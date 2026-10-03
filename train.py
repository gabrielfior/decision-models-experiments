"""train.py — the EDITABLE Tier B script: LoRA on the torso + the head from head.py.

Tier B is the expensive tier (20–25 min on a 3090 / A10G for 8k rows, 1 epoch).
The agent may edit this file to change: which torso, the LoRA rank/alpha/targets,
which layers get adapters, the optimiser and schedule, and the laddered depth heads.
It may NOT change the data split, the scorer, or the keep rule; those come from prepare.py.

Physics reading of what happens here. The torso's weight matrices W are frozen.
LoRA adds a rank-r perturbation W -> W + B A (A: r x d, B: d x r). Only A, B and the
head are trained, so the perturbation has rank r by construction, touches ~1–2% as
many numbers as W, and the model cannot drift far from its pretraining. The head
reads the perturbed evolution's state at <decide> and at each </opt> and scores.

Baseline hyperparameters are Kev's 0.8B recipe (runs/q35-08b/02-trial-2/provenance.json)
with Strands' separate head learning rate, except that we run 1 epoch over 8k rows
instead of Kev's 2 epochs over 12.5k, for cost.

Distillation (--kl-teacher data/teacher/<name>.jsonl --kl-weight a): scripts/teacher.py writes a teacher's option
probabilities per row id in canonical order; the loss becomes (1 - a) CE + a KL(teacher || student) over the valid
options, with the teacher vector permuted into the presented option order. Rows the file misses stay on CE.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import head as head_mod
import prepare as P

# ---- editable defaults ---------------------------------------------------------------
TORSO = "Qwen/Qwen3.5-0.8B-Base"
LORA = dict(
    r=16,
    lora_alpha=32,            # Kev: alpha = 2r
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",            # attention
        "gate_proj", "up_proj", "down_proj",               # MLP
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",  # GatedDeltaNet
    ],
    layers_to_transform=None,  # None = all layers; e.g. list(range(12, 24)) = top half only
)
LR, HEAD_LR, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1e-3, 0.01, 1.0
EPOCHS, BATCH, WARMUP_FRAC = 1, 8, 0.1
MICRO_BATCH = 4               # rows per forward pass; BATCH // MICRO_BATCH gradient-accumulation steps make up a batch of 8.
                              # 8 rows of banking77 (77 options, ~800 tokens) overflow a 24 GB card even with checkpointing.
DEPTH_HEADS = False           # True = laddered heads at every TAP_LAYER, summed loss (Needle)
# --------------------------------------------------------------------------------------


def apply_head_overrides(spec: str | None) -> None:
    """--head-cfg '{"name": "cross_option", "residual_pointer": true}' changes only the given HEAD_CONFIG keys."""
    if spec:
        head_mod.HEAD_CONFIG.update(json.loads(spec))


def split_plan(train_splits: str, calib_split: str) -> tuple[list[str], str, list[str]]:
    """Which splits are trained on, which one fits the temperatures (and is reported as "dev"),
    and which remaining splits are reported as held-out. Final retrain: --train-splits train,dev --calib-split heldout."""
    tr = [x for x in train_splits.split(",") if x]
    report = [x for x in ("train", "dev", "heldout") if x not in tr and x != calib_split]
    return tr, calib_split, report


def head_configs(n_layers: int, depth_heads: bool) -> list[dict]:
    """The exact config of every head this run trains: one per tap for laddered heads, else one.
    Saved to config.json so serve.py rebuilds the head that was trained, not the current default."""
    if depth_heads:
        return [{**head_mod.HEAD_CONFIG, "layer": i} for i in range(n_layers)]
    return [dict(head_mod.HEAD_CONFIG)]


def gather_readout(hidden_states, decide_pos, opt_pos, opt_mask, taps=P.TAP_LAYERS):
    """From the torso's per-layer states pick the vectors the head reads.

    hidden_states: tuple of [B, T, d] (one per layer, index 0 = embeddings)
    decide_pos: [B] position of <decide>;  opt_pos: [B, K] position of each </opt>
    returns h_ans [B, L, d], h_opts [B, L, K, d] for L = len(P.TAP_LAYERS)
    """
    hs = torch.stack([hidden_states[l] for l in taps], dim=1)           # [B, L, T, d]
    B, L, T, d = hs.shape
    bi = torch.arange(B, device=hs.device)
    h_ans = hs[bi, :, decide_pos]                                        # [B, L, d]
    safe_opt = opt_pos.clamp(min=0)
    h_opts = hs[bi[:, None], :, safe_opt]                                # [B, K, L, d]
    h_opts = h_opts.permute(0, 2, 1, 3).contiguous()                     # [B, L, K, d]
    h_opts = h_opts * opt_mask[:, None, :, None]
    return h_ans, h_opts


def forward_logits(torso, heads, batch, taps=P.TAP_LAYERS):
    out = torso(input_ids=batch["ids"], attention_mask=batch["attn"], output_hidden_states=True)
    h_ans, h_opts = gather_readout(out.hidden_states, batch["decide_pos"], batch["opt_pos"], batch["opt_mask"], taps)
    h_ans, h_opts = h_ans.float(), h_opts.float()
    # The head runs in fp32 in both training and eval (the torso may be under bf16 autocast).
    with torch.autocast(device_type=h_ans.device.type, enabled=False):
        return [h(h_ans, h_opts, batch["opt_mask"]) for h in heads]


# ---- teacher distillation (--kl-teacher; the file comes from scripts/teacher.py) -------------
def load_teacher(path) -> dict[str, list[float]]:
    """data/teacher/<name>.jsonl ({"id", "probs"} per line; probs in CANONICAL option order) -> {id: probs}."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = [float(x) for x in r["probs"]]
    return out


def teacher_batch(teacher: dict, ids: list[str], perms: list[list[int]], K: int, device: str = "cpu"):
    """Teacher probabilities in PRESENTED order, padded to K: presented position j shows canonical option perm[j],
    so t_presented[j] = t_canonical[perm[j]].  Returns ([B, K] float, [B] bool: row found in the teacher file)."""
    t = torch.zeros(len(ids), K)
    has = torch.zeros(len(ids), dtype=torch.bool)
    for i, (rid, perm) in enumerate(zip(ids, perms)):
        p = teacher.get(rid)
        if p is None or len(p) != len(perm):
            continue
        t[i, : len(perm)] = torch.tensor([p[j] for j in perm], dtype=torch.float32)
        has[i] = True
    return t.to(device), has.to(device)


def kl_to_teacher(logits: torch.Tensor, teacher_presented: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over the valid options, mean over rows (F.kl_div batchmean).
    The student distribution is softmax over the valid options only; padded options carry no teacher mass and no term."""
    mask = opt_mask.bool()
    log_s = F.log_softmax(logits.float().masked_fill(~mask, -1e4), dim=-1)      # finite fill: 0 * (-1e4) stays 0 below
    t = teacher_presented.float() * mask
    per = F.kl_div(log_s, t, reduction="none") * mask                             # t * (log t - log s), 0 where t = 0
    return per.sum() / logits.shape[0]


def distill_loss(head, logits, labels, teacher_presented, has_teacher, opt_mask, alpha: float) -> torch.Tensor:
    """(1 - alpha) * head.loss + alpha * KL(teacher || student) on rows the teacher file covers; the head's own loss on
    the others; combined as a mean over rows (exact when head.loss is a per-row mean, as CE is)."""
    has = has_teacher.bool()
    B, n_t = logits.shape[0], int(has.sum())
    if n_t == 0 or alpha <= 0:
        return head.loss(logits, labels)
    kl = kl_to_teacher(logits[has], teacher_presented[has], opt_mask[has])
    l_t = (1 - alpha) * head.loss(logits[has], labels[has]) + alpha * kl
    if n_t == B:
        return l_t
    l_n = head.loss(logits[~has], labels[~has])
    return (n_t * l_t + (B - n_t) * l_n) / B


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="one line: hypothesis for this run")
    ap.add_argument("--torso", default=TORSO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: truncate the train split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=None, help="LoRA rank override (alpha stays 2r)")
    ap.add_argument("--top-half-only", action="store_true", help="adapt only the top half of the layers")
    ap.add_argument("--depth-heads", action="store_true", help="laddered heads at every tap, summed loss (Needle)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-cfg", default=None, help='JSON overrides for head.HEAD_CONFIG, e.g. {"name":"cross_option"}')
    ap.add_argument("--train-splits", default="train", help="comma list of splits to train on (final retrain: train,dev)")
    ap.add_argument("--calib-split", default="dev", help="split that fits temperatures and is reported as dev")
    ap.add_argument("--kl-teacher", default=None, help="data/teacher/<name>.jsonl from scripts/teacher.py: distil from its option distributions")
    ap.add_argument("--kl-weight", type=float, default=0.5, help="alpha in (1-alpha)*CE + alpha*KL(teacher||student); rows missing from the teacher file use CE")
    args = ap.parse_args(argv)
    apply_head_overrides(args.head_cfg)
    global DEPTH_HEADS, LR
    if args.rank:
        LORA["r"], LORA["lora_alpha"] = args.rank, 2 * args.rank
    if args.lr:
        LR = args.lr
    if args.depth_heads:
        DEPTH_HEADS = True

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out or P.ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # bf16 weights on GPU (1.5 GB for 0.8B) and gradient checkpointing: activations are recomputed
    # in the backward pass instead of stored, trading ~30% compute for fitting a 24 GB card.
    tcfg = P.torso_config(args.torso)
    markers, taps = tcfg["markers"], tcfg["tap_layers"]
    tok, torso, d_model = P.load_torso(args.torso, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device)
    for p in torso.parameters():
        p.requires_grad_(False)
    if device == "cuda":
        torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    LORA["target_modules"] = tcfg["lora_targets"]      # per-torso module names (Qwen names would not match LFM/ModernBERT)
    if args.top_half_only:
        LORA["layers_to_transform"] = list(range(tcfg["n_layers"] // 2, tcfg["n_layers"]))
    lora_cfg = {k: v for k, v in LORA.items() if v is not None}
    torso = get_peft_model(torso, LoraConfig(**lora_cfg))
    if device == "cuda":
        torso.enable_input_require_grads()

    n_layers = len(taps)
    head_cfgs = head_configs(n_layers, DEPTH_HEADS)
    heads = torch.nn.ModuleList([head_mod.build_head(d_model, n_layers, c) for c in head_cfgs]).to(device)
    # from_pretrained leaves the model in eval mode and peft keeps it there: without this, LoRA dropout
    # is a no-op and HF gradient checkpointing (which checks self.training) never activates.
    torso.train()
    heads.train()

    tr_splits, calib_split, report_splits = split_plan(args.train_splits, args.calib_split)
    train_rows = [r for sname in tr_splits for r in P.load_split(sname)][: args.max_rows]
    dev_rows = P.load_split(calib_split)
    heldout_rows = [r for sname in report_splits for r in P.load_split(sname)]
    print(f"train on {tr_splits} ({len(train_rows)} rows); calibrate/select on {calib_split} ({len(dev_rows)}); report {report_splits} ({len(heldout_rows)})", flush=True)
    teacher = None
    if args.kl_teacher:
        teacher = load_teacher(args.kl_teacher)
        covered = sum(r["id"] in teacher for r in train_rows)
        print(f"teacher {args.kl_teacher}: {len(teacher)} rows, covers {covered}/{len(train_rows)} train rows; alpha {args.kl_weight}", flush=True)

    lora_params = [p for p in torso.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": LR}, {"params": heads.parameters(), "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    accum = max(1, BATCH // MICRO_BATCH)
    steps = args.epochs * math.ceil(math.ceil(len(train_rows) / MICRO_BATCH) / accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, HEAD_LR], total_steps=steps, pct_start=WARMUP_FRAC)

    rng = np.random.default_rng(args.seed)   # numpy: encode_row shuffles options with it
    t0 = time.time()
    step = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for mb, batch in enumerate(P.batches(tok, train_rows, MICRO_BATCH, shuffle_options=True, rng=rng, device=device, markers=markers, shuffle_rows=True)):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits_per_head = forward_logits(torso, heads, batch, taps)
                if teacher is None:
                    loss = sum(h.loss(z, batch["label"]) for h, z in zip(heads, logits_per_head)) / accum
                else:
                    t_pres, has_t = teacher_batch(teacher, batch["id"], batch["perm"], batch["opt_mask"].shape[1], device)
                    loss = sum(distill_loss(h, z, batch["label"], t_pres, has_t, batch["opt_mask"], args.kl_weight)
                               for h, z in zip(heads, logits_per_head)) / accum
            loss.backward()
            if (mb + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_([*lora_params, *heads.parameters()], GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item() * accum:.4f} {time.time() - t0:.0f}s", flush=True)
        # a leftover partial accumulation at the end of an epoch is applied, not carried into the next epoch
        if any(p.grad is not None for p in heads.parameters()):
            torch.nn.utils.clip_grad_norm_([*lora_params, *heads.parameters()], GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
    train_s = time.time() - t0

    # ---- evaluation: frozen harness, two option orders, per-type temperature on dev ------
    torso.eval()
    heads.eval()
    final_head = heads[-1]

    def predict(rows, order):
        return P.predict_rows(lambda b: forward_logits(torso, [final_head], b, taps)[0], tok, rows, order=order, device=device, markers=markers)

    dev_a = predict(dev_rows, order="identity")
    dev_b = predict(dev_rows, order="reversed")
    temps = P.fit_temperature_by_type(dev_a)
    report = P.evaluate(dev_a, dev_b, predict(heldout_rows, order="identity"), temps)
    report.update(note=args.note, tier="B", seeds=1, train_s=train_s, torso=args.torso, run_dir=str(run_dir),
                  per_seed=[report["dev_selection"]], train_splits=tr_splits, calib_split=calib_split)
    # Tier B gate: same rule as Tier A, against the latest kept Tier B row (single seed -> dev SE binds).
    base = P.read_baseline(tier="B")
    report["baseline"] = base
    report["kept"] = True if base is None else P.keep(base, report["per_seed"], report["dev_se"])
    P.append_results(report)
    torso.save_pretrained(run_dir / "lora")
    torch.save(final_head.state_dict(), run_dir / "head.pt")
    (run_dir / "config.json").write_text(json.dumps({"lora": LORA, "lr": LR, "head_lr": HEAD_LR,
                                                       "epochs": args.epochs, "batch": BATCH, "micro_batch": MICRO_BATCH,
                                                       "head": head_cfgs[-1], "heads": head_cfgs, "temps": temps, "torso": args.torso, "tap_layers": list(taps),
                                                       "kl_teacher": args.kl_teacher, "kl_weight": args.kl_weight if args.kl_teacher else None,
                                                       "report": report}, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
