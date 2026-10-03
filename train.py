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
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

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
DEPTH_HEADS = False           # True = laddered heads at every TAP_LAYER, summed loss (Needle)
# --------------------------------------------------------------------------------------


def gather_readout(hidden_states, decide_pos, opt_pos, opt_mask):
    """From the torso's per-layer states pick the vectors the head reads.

    hidden_states: tuple of [B, T, d] (one per layer, index 0 = embeddings)
    decide_pos: [B] position of <decide>;  opt_pos: [B, K] position of each </opt>
    returns h_ans [B, L, d], h_opts [B, L, K, d] for L = len(P.TAP_LAYERS)
    """
    hs = torch.stack([hidden_states[l] for l in P.TAP_LAYERS], dim=1)   # [B, L, T, d]
    B, L, T, d = hs.shape
    bi = torch.arange(B, device=hs.device)
    h_ans = hs[bi, :, decide_pos]                                        # [B, L, d]
    safe_opt = opt_pos.clamp(min=0)
    h_opts = hs[bi[:, None], :, safe_opt]                                # [B, K, L, d]
    h_opts = h_opts.permute(0, 2, 1, 3).contiguous()                     # [B, L, K, d]
    h_opts = h_opts * opt_mask[:, None, :, None]
    return h_ans, h_opts


def forward_logits(torso, heads, batch):
    out = torso(input_ids=batch["ids"], attention_mask=batch["attn"], output_hidden_states=True)
    h_ans, h_opts = gather_readout(out.hidden_states, batch["decide_pos"], batch["opt_pos"], batch["opt_mask"])
    if DEPTH_HEADS:
        # one head per depth; each sees only its own layer (set via cfg layer index)
        return [h(h_ans, h_opts, batch["opt_mask"]) for h in heads]
    return [heads[0](h_ans, h_opts, batch["opt_mask"])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="one line: hypothesis for this run")
    ap.add_argument("--torso", default=TORSO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: truncate the train split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out or P.ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    tok, torso, d_model = P.load_torso(args.torso, dtype=torch.float32, device=device)
    for p in torso.parameters():
        p.requires_grad_(False)
    lora_cfg = {k: v for k, v in LORA.items() if v is not None}
    torso = get_peft_model(torso, LoraConfig(**lora_cfg))

    n_layers = len(P.TAP_LAYERS)
    if DEPTH_HEADS:
        heads = torch.nn.ModuleList([head_mod.build_head(d_model, n_layers, {"layer": i}) for i in range(n_layers)])
    else:
        heads = torch.nn.ModuleList([head_mod.build_head(d_model, n_layers)])
    heads.to(device)

    train_rows = P.load_split("train")[: args.max_rows]
    dev_rows = P.load_split("dev")
    heldout_rows = P.load_split("heldout")

    lora_params = [p for p in torso.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": LR}, {"params": heads.parameters(), "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    steps = args.epochs * math.ceil(len(train_rows) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, HEAD_LR], total_steps=steps, pct_start=WARMUP_FRAC)

    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in P.batches(tok, train_rows, BATCH, shuffle_options=True, rng=rng, device=device):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits_per_head = forward_logits(torso, heads, batch)
                loss = sum(h.loss(z, batch["label"]) for h, z in zip(heads, logits_per_head))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*lora_params, *heads.parameters()], GRAD_CLIP)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0

    # ---- evaluation: frozen harness, two option orders, per-type temperature on dev ------
    torso.eval()
    heads.eval()
    final_head = heads[-1]

    def predict(rows, order):
        return P.predict_rows(lambda b: forward_logits(torso, [final_head], b)[0], tok, rows, order=order, device=device)

    dev_a = predict(dev_rows, order="identity")
    dev_b = predict(dev_rows, order="reversed")
    temps = P.fit_temperature_by_type(dev_a)
    report = P.evaluate(dev_a, dev_b, predict(heldout_rows, order="identity"), temps)
    report.update(note=args.note, tier="B", seeds=1, train_s=train_s, torso=args.torso, run_dir=str(run_dir))
    P.append_results(report)
    torso.save_pretrained(run_dir / "lora")
    torch.save(final_head.state_dict(), run_dir / "head.pt")
    (run_dir / "config.json").write_text(json.dumps({"lora": LORA, "lr": LR, "head_lr": HEAD_LR,
                                                       "epochs": args.epochs, "batch": BATCH,
                                                       "head": head_mod.HEAD_CONFIG, "temps": temps,
                                                       "report": report}, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
