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


class Decider:
    """Torso (+ optional LoRA) + head + per-type temperatures, loaded once."""

    def __init__(self, torso: str, run_dir: Path | None, head_path: Path | None, device: str):
        import torch
        import head as head_mod
        import train
        self.device = device
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
        self.head = head_mod.build_head(d, len(P.TAP_LAYERS), cfg.get("head")).to(device)
        if head_path is not None:
            self.head.load_state_dict(torch.load(head_path, map_location=device))
        self.head.eval()
        self.train = train

    def logits(self, batch):
        import torch
        with torch.no_grad():
            out = self.model(input_ids=batch["ids"], attention_mask=batch["attn"], output_hidden_states=True)
            h_ans, h_opts = self.train.gather_readout(out.hidden_states, batch["decide_pos"], batch["opt_pos"], batch["opt_mask"])
            return self.head(h_ans.float(), h_opts.float(), batch["opt_mask"])

    def decide(self, state, question: dict) -> dict:
        row = P.question_to_row(state, question)
        batch = next(P.batches(self.tok, [row], 1, shuffle_options=False, device=self.device))
        z = self.logits(batch)[0, : row["n_options"]].float().cpu().numpy()
        probs = P._softmax(z / self.temps.get(row["qtype"], 1.0))
        return build_answer(row["qtype"], row["options"], probs)


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
    ap.add_argument("--run", default=None, help="run dir with lora/, head.pt, config.json")
    ap.add_argument("--head", default=None, help="a head.pt to use (Tier A head on the frozen torso)")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--name", default="decider-autoresearch")
    a = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Decider(a.torso, a.run, a.head, device)
    print(f"serving {a.name} on http://127.0.0.1:{a.port}/v1/systemone (device {device}, temps {d.temps})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(d, a.name)).serve_forever()


if __name__ == "__main__":
    main()
