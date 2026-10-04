"""Export a trained decider as a small, self-contained model directory.

Three size reductions, each checked to leave the decision logits unchanged (or nearly so):

  1. merge   - fold the LoRA deltas into the torso weights (no adapter at inference);
  2. truncate - keep only the layers the head reads (HEAD_CONFIG["layer"] = 1 reads the tap at layer 16 of 24,
                so layers 17-24 never influence a decision: dropping them is exact);
  3. trim    - keep the first VOCAB_CUT vocabulary ids (byte-level BPE assigns ids in merge order, so these are the
                most frequent pieces) plus the added special tokens; rarer pieces are re-split by the remaining
                merges. Measured on our corpus and JevBench, ids >= 98304 carry ~0.2% of token occurrences.

Usage:
  uv run python scripts/export.py --run <run dir> --out <dir> [--torso Qwen/Qwen3.5-2B-Base] [--keep-layers 16]
                                  [--vocab-cut 98304 | --no-trim] [--dtype bf16]
The output has  <out>/torso/ (HF model + tokenizer + decider_torso.json), <out>/head.pt, <out>/config.json
so that  serve.py --torso <out>/torso --run <out>  serves it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare as P  # noqa: E402

VOCAB_CUT = 98304
TORSO_META = "decider_torso.json"


def trim_tokenizer_json(tj: dict, cut: int) -> tuple[dict, list[int]]:
    """Return (new tokenizer.json dict, old ids of the kept rows in new-id order).

    Vocabulary ids < cut keep their id; added (special) tokens are renumbered to cut, cut+1, ... in their old
    order; merges whose parts or result fall outside the kept vocabulary are dropped.
    """
    model = tj["model"]
    vocab = model["vocab"]
    kept = {t: i for t, i in vocab.items() if i < cut}
    merges = []
    for m in model["merges"]:
        a, b = (m if isinstance(m, list) else m.split(" ", 1))
        if a in kept and b in kept and (a + b) in kept:
            merges.append(m)
    added = sorted(tj["added_tokens"], key=lambda a: a["id"])
    old_ids = list(range(cut)) + [a["id"] for a in added]
    new_added = []
    for j, a in enumerate(added):
        new_added.append({**a, "id": cut + j})
    new = dict(tj)
    new["model"] = {**model, "vocab": kept, "merges": merges}
    new["added_tokens"] = new_added
    return new, old_ids


def write_trimmed_tokenizer(src_tok, out_dir: Path, cut: int) -> list[int]:
    """Save a trimmed copy of a fast tokenizer to out_dir; returns the old ids kept, in new-id order."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    tj = json.loads(src_tok.backend_tokenizer.to_str())
    new_tj, old_ids = trim_tokenizer_json(tj, cut)
    tk = Tokenizer.from_str(json.dumps(new_tj))
    specials = {k: v for k, v in src_tok.special_tokens_map.items() if k != "additional_special_tokens"}
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, model_max_length=src_tok.model_max_length,
                                   additional_special_tokens=src_tok.special_tokens_map.get("additional_special_tokens", []),
                                   **specials)
    fast.save_pretrained(out_dir)
    return old_ids


def truncate_torso(model, keep_layers: int):
    """Keep the first keep_layers blocks and drop the final norm, so hidden_states[keep_layers] is the raw
    output of block keep_layers exactly as in the full model (the final norm would otherwise be applied to it)."""
    import torch
    model.layers = model.layers[:keep_layers]
    model.norm = torch.nn.Identity()
    model.config.num_hidden_layers = keep_layers
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = list(model.config.layer_types)[:keep_layers]
    return model


def trim_embeddings(model, old_ids: list[int]):
    import torch
    emb = model.get_input_embeddings()
    rows = emb.weight.data[torch.as_tensor(old_ids, device=emb.weight.device)]
    new = torch.nn.Embedding(len(old_ids), rows.shape[1], padding_idx=None, dtype=rows.dtype, device=rows.device)
    new.weight.data.copy_(rows)
    model.set_input_embeddings(new)
    model.config.vocab_size = len(old_ids)
    for k in ("pad_token_id", "eos_token_id", "bos_token_id"):
        v = getattr(model.config, k, None)
        if isinstance(v, int) and v in old_ids:
            setattr(model.config, k, old_ids.index(v))
    return model


def export(run: Path, out: Path, torso: str, keep_layers: int, vocab_cut: int | None, dtype_name: str = "bf16"):
    import torch
    from peft import PeftModel
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    tcfg = P.torso_config(torso)
    tok, model, d_model = P.load_torso(torso, dtype=dtype, device="cpu")
    if (run / "lora").exists():
        model = PeftModel.from_pretrained(model, run / "lora").merge_and_unload()
    n_before = sum(p.numel() for p in model.parameters())
    model = truncate_torso(model, keep_layers)
    out_torso = out / "torso"
    out_torso.mkdir(parents=True, exist_ok=True)
    if vocab_cut:
        old_ids = write_trimmed_tokenizer(tok, out_torso, vocab_cut)
        model = trim_embeddings(model, old_ids)
    else:
        tok.save_pretrained(out_torso)
    # The final norm is dropped: the saved config must tell load_torso to do the same.
    model.config.tie_word_embeddings = False     # no LM head in this artefact; nothing to tie
    model.save_pretrained(out_torso, safe_serialization=True)
    # per-torso meta so prepare.torso_config() understands the exported directory
    taps = tuple(min(t, keep_layers) for t in tcfg["tap_layers"])
    meta = {"base": torso, "markers": tcfg["markers"], "tap_layers": list(taps), "n_layers": keep_layers,
            "skip_final_norm": True, "lora_targets": tcfg["lora_targets"], "vocab_cut": vocab_cut,
            "params": sum(p.numel() for p in model.parameters()), "params_before_export": n_before}
    (out_torso / TORSO_META).write_text(json.dumps(meta, indent=2))
    shutil.copy(run / "head.pt", out / "head.pt")
    cfg = json.loads((run / "config.json").read_text())
    cfg.update(torso=str(out_torso), tap_layers=list(taps), exported_from=str(run), export=meta)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--torso", default=None, help="base torso; default: the run's config.json 'torso'")
    ap.add_argument("--keep-layers", type=int, default=None, help="default: the tap the head reads (HEAD_CONFIG layer index)")
    ap.add_argument("--vocab-cut", type=int, default=VOCAB_CUT)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    torso = a.torso or cfg["torso"]
    taps = cfg.get("tap_layers") or list(P.torso_config(torso)["tap_layers"])
    keep = a.keep_layers or taps[cfg["head"].get("layer", -1)]
    meta = export(a.run, a.out, torso, keep, None if a.no_trim else a.vocab_cut, a.dtype)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
