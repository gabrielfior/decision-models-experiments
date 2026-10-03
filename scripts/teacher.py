"""scripts/teacher.py — teacher option distributions for distillation (consumed by train.py --kl-teacher).

For every row of the given split(s) the script writes one line  {"id": <row id>, "probs": [...]}  with the teacher's
probability vector in CANONICAL option order (the order of row["options"], the same order serve.canonical_probs and
prepare.predict_rows map back to) to  data/teacher/<name>.jsonl.  train.py then permutes each vector into the order
the options were presented in and minimises (1 - a) CE + a KL(teacher || student).

Two teacher backends:

  ours     a run dir (or several: an ensemble) of THIS repo, read through serve.Decider with test-time averaging over
           --tta option orders.  Self-distillation / ensemble distillation; no new model code.
               uv run python scripts/teacher.py --backend ours --run runs/<dir> [--run runs/<dir2>] --tta 4 --name ours-tta4

  mapika   the open model Mapika/decider-4b (https://huggingface.co/Mapika/decider-4b, code https://github.com/Mapika/decider):
           Qwen3.5-4B-Base fine-tuned to answer typed questions by emitting an option letter at an answer slot.  Its public
           inference format (decider/prompt.py build(), decider/model.py slot_logits(), decider/systemone.py render_question()
           and isolated_rows(), decider/temperature.py; decider_config.json of the 4b repo) is reproduced here with plain
           transformers, in one place (the MAPIKA_* constants and the mapika_* functions below):

             Context:\\n<state>\\n\\nQuestion: <instructions>\\nOptions:\\n(A) <opt>\\n(B) <opt>...\\nAnswer: (

           - options render as render_question does: choice "key" or "key: description"; score "i: description";
             noul "no" / "yes" or "no: <false desc>" / "yes: <true desc>".
           - up to 10 options use the "(A)".."(J)" text rendering; above 10 ("wide") every option gets one single-token label
             (A..Z, then two-letter upper-case strings that are one token), built from token ids: "\\n(" + <label> + ") text".
           - the probability is read at the last token ("(" of "Answer: ("): the last hidden state projected onto the LM-head
             rows of the label tokens, softmax over the first n labels at the per-type temperature of decider_config.json
             (4b: choice 1.11, noul 1.56, score 1.287; fallback "temperature" 1.099).
           - decider-4b has "isolated_levels": true: a score question is NOT asked as a list.  Each level becomes its own
             yes/no row  "<instructions>\\nProposed answer: <level description>\\nDoes the proposed answer fit?"  with options
             ["no", "yes"], read at the score temperature; the P(yes) of the levels are normalised into the level distribution.
             --score-list asks the list form instead.
               uv run python scripts/teacher.py --backend mapika --tta 2 --name mapika

Both backends average over --tta option orders for choice questions (identity, reversed, cyclic shifts; serve.tta_orders),
which makes the teacher (nearly) order-invariant.  noul and score keep their order because it carries meaning.
GPU host only (the 4b teacher wants ~10 GB in bf16); nothing here runs in the test suite except the pure functions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare as P   # noqa: E402
import serve          # noqa: E402

TEACHER_DIR = ROOT / "data" / "teacher"


# ---------------------------------------------------------------------------------------------
# common: option orders, the output file
# ---------------------------------------------------------------------------------------------
def perm_for(order: str, n: int, qtype: str) -> list[int]:
    """Presented position j shows canonical option perm[j]; the same orders as prepare.encode_row(order=...).
    Only choice questions are permuted: no/yes and ordinal levels keep their order."""
    perm = list(range(n))
    if qtype != "choice":
        return perm
    if order == "reversed":
        return perm[::-1]
    if order.startswith("shift:"):
        k = int(order.split(":")[1]) % n
        return perm[k:] + perm[:k]
    return perm


def write_teacher(path: Path, items) -> int:
    """items: iterable of (row id, canonical probability vector). One JSON object per line. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for rid, probs in items:
            p = np.asarray(probs, dtype=np.float64)
            f.write(json.dumps({"id": rid, "probs": [float(x) for x in p]}) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------------------------
# backend "ours": a run dir of this repo through serve.Decider
# ---------------------------------------------------------------------------------------------
class OursTeacher:
    def __init__(self, torso: str, runs: list[str], device: str, tta: int):
        runs = runs or []
        if not runs:
            raise SystemExit("--backend ours needs at least one --run <dir>")
        self.decider = serve.Decider(torso, Path(runs[0]), None, device, tta=tta, extra_runs=[Path(r) for r in runs[1:]])

    def probs_many(self, rows: list[dict]) -> list[np.ndarray]:
        return [serve.average_probs(self.decider.probs(r)) for r in rows]


# ---------------------------------------------------------------------------------------------
# backend "mapika": Mapika/decider-4b with plain transformers
# ---------------------------------------------------------------------------------------------
MAPIKA_REPO = "Mapika/decider-4b"
# decider_config.json of Mapika/decider-4b (version 4b-v2.1, 2026-09-24); used when the Hub file cannot be fetched.
MAPIKA_DEFAULT_CONFIG = {"temperature": 1.099, "temperature_by_type": {"choice": 1.11, "noul": 1.56, "score": 1.287},
                         "isolated_levels": True, "neutralize_none": False, "layout": "plain"}
MAPIKA_LETTERS = "ABCDEFGHIJ"      # <= 10 options: the "(A) .." text rendering
MAPIKA_NARROW = len(MAPIKA_LETTERS)
MAPIKA_MAX_OPTIONS = 255           # wide rendering: one single-token label per option
MAPIKA_CONTEXT = "Context:\n"
MAPIKA_QUESTION = "\n\nQuestion: {q}\nOptions:"
MAPIKA_OPTION = "\n({letter}) {text}"
MAPIKA_ANSWER = "\nAnswer: ("
MAPIKA_ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"


def mapika_temperatures(cfg: dict | None) -> dict[str, float]:
    """{qtype: T} from a decider_config.json dict: temperature_by_type[t], else temperature. None = the published 4b values."""
    cfg = MAPIKA_DEFAULT_CONFIG if cfg is None else cfg
    base = float(cfg.get("temperature", 1.0))
    by_type = cfg.get("temperature_by_type") or {}
    return {t: float(by_type.get(t, base)) for t in P.QTYPES}


def mapika_option_texts(row: dict) -> list[str]:
    """Option strings in canonical order, as decider.systemone.render_question renders Jev-shaped criteria:
    choice "key" / "key: description", score "i: description", noul "no" / "yes" / "no: <false desc>" / "yes: <true desc>"."""
    return [f"{key}: {desc}" if desc not in (None, "") else str(key) for key, desc in row["options"]]


def mapika_prompt_text(state: str, question: str, options: list[str]) -> str:
    """The plain state-first prompt for up to 10 options (presented order), as decider.prompt.build renders it."""
    assert len(options) <= MAPIKA_NARROW, "more than 10 options need the wide, id-level rendering (mapika_encode)"
    body = "".join(MAPIKA_OPTION.format(letter=MAPIKA_LETTERS[j], text=o) for j, o in enumerate(options))
    return MAPIKA_CONTEXT + state + MAPIKA_QUESTION.format(q=question) + body + MAPIKA_ANSWER


def _strip_level_number(text: str) -> str:
    import re
    return re.sub(r"^\s*-?\d+\s*:\s*", "", text)


def mapika_isolated_rows(row: dict) -> list[tuple[str, list[str]]]:
    """A score question with isolated levels: one yes/no question per level, the level shown without its number."""
    return [(MAPIKA_ISOLATED.format(q=row["instructions"], level=_strip_level_number(desc if desc not in (None, "") else str(key))),
             ["no", "yes"]) for key, desc in row["options"]]


def combine_isolated(p_yes) -> list[float]:
    """Per-level P(fits) -> a distribution over levels (decider.systemone.combine_isolated)."""
    p = [float(x) for x in p_yes]
    tot = sum(p) or 1e-9
    return [x / tot for x in p]


_LABELS: dict = {}


def mapika_label_table(tok) -> tuple[list[str], list[int], list[int]]:
    """(label strings, label token ids, ids of "\\n("): A..Z then two-letter upper-case strings that tokenize to one token,
    up to 255. The first ten are A..J, so narrow prompts read the same letters the text rendering shows."""
    key = id(tok)
    if key not in _LABELS:
        import string
        U = string.ascii_uppercase
        out = []
        for name in list(U) + [a + b for a in U for b in U]:
            t = tok.encode(name, add_special_tokens=False)
            if len(t) == 1:
                out.append((name, t[0]))
            if len(out) == MAPIKA_MAX_OPTIONS:
                break
        _LABELS[key] = (tok, [n for n, _ in out], [i for _, i in out], tok.encode("\n(", add_special_tokens=False))
    _, names, ids, open_ids = _LABELS[key]
    return names, ids, open_ids


def mapika_encode(tok, row: dict, perm: list[int], max_ctx_tokens: int = 1536, question: str | None = None,
                  options: list[str] | None = None) -> dict:
    """Token ids of one single-question prompt in the plain state-first layout, the answer-slot position and the label
    token ids of its options.  `options` (canonical order) default to mapika_option_texts(row); `perm` presents them in
    the given order.  question/options can be overridden for the isolated-level rows."""
    opts_canon = mapika_option_texts(row) if options is None else list(options)
    opts = [opts_canon[j] for j in perm]
    q = row["instructions"] if question is None else question
    n = len(opts)
    names, lab_ids, open_ids = mapika_label_table(tok)
    if n > len(lab_ids):
        raise ValueError(f"{n} options but only {len(lab_ids)} single-token labels")
    ctx_ids = tok.encode(MAPIKA_CONTEXT + row["state"], add_special_tokens=False)[:max_ctx_tokens]
    if n <= MAPIKA_NARROW:
        piece = tok.encode(MAPIKA_QUESTION.format(q=q) + "".join(MAPIKA_OPTION.format(letter=MAPIKA_LETTERS[j], text=o) for j, o in enumerate(opts))
                           + MAPIKA_ANSWER, add_special_tokens=False)
    else:
        piece = tok.encode(MAPIKA_QUESTION.format(q=q), add_special_tokens=False)
        for j, o in enumerate(opts):
            piece += open_ids + [lab_ids[j]] + tok.encode(f") {o}", add_special_tokens=False)
        piece += tok.encode(MAPIKA_ANSWER, add_special_tokens=False)
    ids = list(ctx_ids) + piece
    return {"ids": ids, "slot": len(ids) - 1, "n_options": n, "perm": list(perm), "letter_ids": lab_ids[:n]}


class MapikaTeacher:
    """Mapika/decider-4b read with transformers: last hidden state at the answer slot -> LM-head rows of the labels."""

    def __init__(self, repo: str = MAPIKA_REPO, device: str = "cuda", tta: int = 1, batch_size: int = 8,
                 max_ctx_tokens: int = 1536, score_list: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device, self.tta, self.batch_size, self.max_ctx_tokens = device, int(tta), int(batch_size), int(max_ctx_tokens)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.lm = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        self.cfg = self.load_config(repo)
        self.temps = mapika_temperatures(self.cfg)
        self.isolated = bool(self.cfg.get("isolated_levels", False)) and not score_list
        cfg = self.lm.config
        self.softcap = next((float(getattr(c, "final_logit_softcapping")) for c in (cfg, getattr(cfg, "text_config", None))
                             if c is not None and getattr(c, "final_logit_softcapping", None)), None)
        names, ids, _ = mapika_label_table(self.tok)
        for j, L in enumerate(MAPIKA_LETTERS):     # the text rendering's letters must be the label tokens read
            assert self.tok.encode(L, add_special_tokens=False) == [ids[j]], L
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        print(f"mapika teacher {repo}: temps {self.temps}, isolated levels {self.isolated}, {len(ids)} labels", flush=True)

    @staticmethod
    def load_config(repo: str) -> dict:
        try:
            from huggingface_hub import hf_hub_download
            return json.loads(Path(hf_hub_download(repo, "decider_config.json")).read_text())
        except Exception as e:  # offline or no config in the repo: the published 4b values
            print(f"decider_config.json not fetched ({e}); using the published defaults", flush=True)
            return dict(MAPIKA_DEFAULT_CONFIG)

    def _letter_logits(self, encs: list[dict]) -> list[np.ndarray]:
        """One forward pass per batch; returns, per enc, the logits of its n_options label tokens (fp32 numpy)."""
        torch, out = self.torch, []
        base = getattr(self.lm, "model", None) or self.lm.get_decoder()
        W = self.lm.lm_head.weight
        with torch.no_grad():
            for s in range(0, len(encs), self.batch_size):
                chunk = encs[s:s + self.batch_size]
                T = max(len(e["ids"]) for e in chunk)
                ids = torch.full((len(chunk), T), self.pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), T), dtype=torch.long)
                for i, e in enumerate(chunk):
                    ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
                    attn[i, : len(e["ids"])] = 1
                h = base(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
                slots = torch.tensor([e["slot"] for e in chunk], device=h.device)
                hs = h[torch.arange(len(chunk), device=h.device), slots]                       # [b, H]
                for i, e in enumerate(chunk):
                    z = (hs[i : i + 1] @ W[e["letter_ids"]].T).float()[0]                        # [n]
                    if self.softcap:
                        z = torch.tanh(z / self.softcap) * self.softcap
                    out.append(z.cpu().numpy())
        return out

    def probs_many(self, rows: list[dict]) -> list[np.ndarray]:
        """Canonical-order probability vectors for a list of rows (all prompts of the list batched together)."""
        encs, plan = [], []
        for r in rows:
            if r["qtype"] == "score" and self.isolated:
                start = len(encs)
                for q, opts in mapika_isolated_rows(r):
                    encs.append(mapika_encode(self.tok, r, [0, 1], self.max_ctx_tokens, question=q, options=opts))
                plan.append(("iso", start, r["n_options"]))
            else:
                orders = serve.tta_orders(self.tta) if r["qtype"] == "choice" else ["identity"]
                start = len(encs)
                for order in orders:
                    encs.append(mapika_encode(self.tok, r, perm_for(order, r["n_options"], r["qtype"]), self.max_ctx_tokens))
                plan.append(("list", start, len(orders)))
        logits = self._letter_logits(encs)
        out = []
        for r, (kind, start, n) in zip(rows, plan):
            T = self.temps.get(r["qtype"], 1.0)
            if kind == "iso":
                p_yes = [P._softmax(logits[start + j] / T)[1] for j in range(n)]
                out.append(np.asarray(combine_isolated(p_yes)))
            else:
                ps = [serve.canonical_probs(logits[start + j], encs[start + j]["perm"], T) for j in range(n)]
                out.append(serve.average_probs(ps))
        return out


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("ours", "mapika"), required=True)
    ap.add_argument("--rows", default="train", help="comma list of splits to label (default train)")
    ap.add_argument("--name", default=None, help="output file data/teacher/<name>.jsonl (default: the backend)")
    ap.add_argument("--out", default=None, help="explicit output path (overrides --name)")
    ap.add_argument("--tta", type=int, default=1, help="option orders averaged for choice questions")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: truncate the rows")
    ap.add_argument("--chunk", type=int, default=32, help="rows per scoring chunk (mapika batches all their prompts)")
    # ours
    ap.add_argument("--run", action="append", default=None, help="run dir (repeat for an ensemble)")
    ap.add_argument("--torso", default="Qwen/Qwen3.5-0.8B-Base")
    # mapika
    ap.add_argument("--model", default=MAPIKA_REPO)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=1536)
    ap.add_argument("--score-list", action="store_true", help="ask score questions as one list instead of isolated levels")
    args = ap.parse_args(argv)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = [s for s in args.rows.split(",") if s]
    rows = [r for s in splits for r in P.load_split(s)][: args.max_rows]
    out = Path(args.out) if args.out else TEACHER_DIR / f"{args.name or args.backend}.jsonl"
    if args.backend == "ours":
        teacher = OursTeacher(args.torso, args.run, device, args.tta)
    else:
        teacher = MapikaTeacher(args.model, device, args.tta, args.batch_size, args.max_ctx_tokens, args.score_list)
    print(f"{args.backend}: {len(rows)} rows from {splits} -> {out} (device {device}, tta {args.tta})", flush=True)

    t0, items, agree, labelled = time.time(), [], 0, 0
    for s in range(0, len(rows), args.chunk):
        chunk = rows[s:s + args.chunk]
        for r, p in zip(chunk, teacher.probs_many(chunk)):
            items.append((r["id"], p))
            if r["label"] is not None:
                labelled += 1
                agree += int(np.argmax(p) == r["label"])
        if (s // args.chunk) % 10 == 0:
            print(f"  {len(items)}/{len(rows)} rows, teacher acc so far {agree / max(labelled, 1):.3f}, {time.time() - t0:.0f}s", flush=True)
    n = write_teacher(out, items)
    print(json.dumps({"backend": args.backend, "rows": n, "out": str(out), "teacher_acc": agree / max(labelled, 1),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
