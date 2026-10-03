"""serve.py — expose a trained decider on JevBench's /v1/systemone wire format.

The stock JevBench `typesafe` adapter POSTs
    {"state": ..., "model": ..., "questions": {"decision": {"type", "instructions", "criteria"}}}
and expects
    {"answers": {"decision": <answer>}, "usage": {...}, "model": "..."}
where <answer> is
    noul:   {"type": "noul", "noul": p_yes}
    choice: {"type": "choice", "choice": "<key>", "probabilities": {key: p}}
    score:  {"type": "score", "probabilities": {"0": p0, "1": p1, ...}}
Probabilities must sum to 1 within 1e-3 or the harness marks the item invalid.

Usage:
    uv run python serve.py --run runs/<dir>            # LoRA adapter + head.pt + temps from config.json
    uv run python serve.py --head-only --cache-torso   # frozen torso + a Tier A head.pt (no LoRA)
then  scripts/jevbench.sh <run-name>  drives the harness against http://127.0.0.1:8811.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import prepare as P


def canonical_probs(z: np.ndarray, perm: list[int], temperature: float = 1.0) -> np.ndarray:
    """Softmax of logits given in PRESENTED order, returned in canonical option order (canon[perm[j]] = p_j)."""
    p = P._softmax(np.asarray(z, dtype=np.float64) / temperature)
    canon = np.empty_like(p)
    canon[np.asarray(perm)] = p
    return canon


def tta_orders(n: int) -> list[str]:
    """The option orders used for test-time averaging: identity, reversed, then cyclic shifts.
    Averaging over orders makes the decision (nearly) permutation-invariant at inference, which is
    exactly the property the one-pass scorer lacks; cost grows linearly with n."""
    orders = ["identity", "reversed"] + [f"shift:{k}" for k in range(1, max(n, 1))]
    return orders[:n]


def average_probs(ps: list[np.ndarray]) -> np.ndarray:
    """Test-time augmentation: mean of probability vectors from different option orders (renormalised)."""
    m = np.mean(np.stack(ps), axis=0)
    return m / m.sum()


def build_answer(qtype: str, options: list[tuple[str, str | None]], probs: np.ndarray) -> dict:
    """Map a probability vector over our canonical option order onto the harness's answer shape."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    keys = [k for k, _ in options]
    if qtype == "noul":
        return {"type": "noul", "noul": float(p[keys.index("yes")])}
    table = {k: float(v) for k, v in zip(keys, p)}
    if qtype == "choice":
        return {"type": "choice", "choice": keys[int(np.argmax(p))], "probabilities": table}
    if qtype == "score":
        return {"type": "score", "probabilities": table}
    raise ValueError(qtype)


# Keys that newer head.py versions added to HEAD_CONFIG, with the value they implicitly had before.
HEAD_CONFIG_LEGACY_DEFAULTS = {"norm": False}


def head_config_from_run(saved: dict | None) -> dict | None:
    """A saved run's head config, completed with the defaults that applied when it was trained,
    so build_head() does not inherit newer HEAD_CONFIG defaults the checkpoint never had."""
    if saved is None:
        return None
    return {**HEAD_CONFIG_LEGACY_DEFAULTS, **saved}


class Decider:
    """Torso (+ optional LoRA) + head + per-type temperatures, loaded once. `tta` = number of option
    orders averaged (1 = none). Several run dirs form an ensemble: each member is a full model."""

    def __init__(self, torso: str, run_dir: Path | None, head_path: Path | None, device: str, tta: int = 1, extra_runs: list | None = None):
        self.tta = int(tta) if not isinstance(tta, bool) else (2 if tta else 1)
        self.members = [Decider(torso, r, None, device, 1) for r in (extra_runs or [])]
        import torch
        import head as head_mod
        import train
        self.device = device
        tcfg = P.torso_config(torso)
        self.markers, self.taps = tcfg["markers"], tcfg["tap_layers"]
        self.tok, self.model, d = P.load_torso(torso, dtype=torch.float32 if device == "cpu" else torch.bfloat16, device=device)
        self.temps = {t: 1.0 for t in P.QTYPES}
        cfg = {}
        if run_dir is not None:
            run_dir = Path(run_dir)
            cfg = json.loads((run_dir / "config.json").read_text())
            if (run_dir / "lora").exists():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, run_dir / "lora").merge_and_unload()
            head_path = head_path or run_dir / "head.pt"
            self.temps.update(cfg.get("temps", {}))
        self.model.eval()
        head_cfg = cfg.get("head")
        if head_path is not None and run_dir is None and Path(head_path).with_name("head.json").exists():
            side = json.loads(Path(head_path).with_name("head.json").read_text())   # written by prepare.py tier-a --save
            head_cfg = side.get("head", head_cfg)
            self.temps.update(side.get("temps", {}))
        self.head = head_mod.build_head(d, len(self.taps), head_config_from_run(head_cfg)).to(device)
        if head_path is not None:
            self.head.load_state_dict(torch.load(head_path, map_location=device))
        self.head.eval()
        self.train = train

    def logits(self, batch):
        import torch
        with torch.no_grad():
            out = self.model(input_ids=batch["ids"], attention_mask=batch["attn"], output_hidden_states=True)
            h_ans, h_opts = self.train.gather_readout(out.hidden_states, batch["decide_pos"], batch["opt_pos"], batch["opt_mask"], self.taps)
            return self.head(h_ans.float(), h_opts.float(), batch["opt_mask"])

    def probs(self, row: dict) -> list[np.ndarray]:
        T = self.temps.get(row["qtype"], 1.0)
        orders = tta_orders(self.tta) if row["qtype"] == "choice" else ["identity"]
        ps = []
        for order in orders:
            batch = next(P.batches(self.tok, [row], 1, shuffle_options=False, device=self.device, markers=self.markers, order=order))
            z = self.logits(batch)[0, : row["n_options"]].float().cpu().numpy()
            ps.append(canonical_probs(z, batch["perm"][0], T))
        for m in self.members:
            m.tta = self.tta
            ps += m.probs(row)
        return ps

    def decide(self, state, question: dict) -> dict:
        row = P.question_to_row(state, question)
        return build_answer(row["qtype"], row["options"], average_probs(self.probs(row)))


def make_handler(decider: Decider, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.endswith("/v1/systemone"):
                return self._send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            t0 = time.perf_counter()
            try:
                answers = {qid: decider.decide(body.get("state"), q) for qid, q in (body.get("questions") or {}).items()}
            except Exception as e:  # report, don't crash the harness run
                return self._send(400, {"error": str(e)})
            self._send(200, {"answers": answers, "model": model_name,
                             "usage": {"latency_ms": (time.perf_counter() - t0) * 1000}})

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--torso", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--run", action="append", default=None, help="run dir with lora/, head.pt, config.json; repeat for an ensemble")
    ap.add_argument("--head", default=None, help="a head.pt to use (Tier A head on the frozen torso)")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--name", default="decider-autoresearch")
    ap.add_argument("--tta", nargs="?", const=2, default=1, type=int, help="average over N option orders for choice questions (default 1 = off; bare flag = 2)")
    a = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = a.run or [None]
    d = Decider(a.torso, runs[0], a.head, device, tta=a.tta, extra_runs=runs[1:])
    print(f"serving {a.name} on http://127.0.0.1:{a.port}/v1/systemone (device {device}, tta {d.tta}, members {1 + len(d.members)}, temps {d.temps})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(d, a.name)).serve_forever()


if __name__ == "__main__":
    main()
