"""prepare.py — FROZEN harness. The agent may not edit this file.

Everything that could make the test easier lives here, out of the agent's reach:
the data subset and its fixed split, the hidden-state cache, the scorer, the
keep/discard gate, and the Tier A trainer that imports head.py.

Sections:
  1. constants              seeds, split sizes, tapped layers, question types
  2. scoring                score(), keep()            (pure numpy, tested in tests/test_score.py)
  3. data                   download + fixed 8k/2k split with held-out families   (step 2 of the plan)
  4. sequence layout        how a row becomes tokens and which positions we read   (step 2)
  5. feature cache          run the frozen torso once, save read-out vectors       (step 4)
  6. Tier A trainer         train head.py on the cache, 3 seeds, append results.tsv (step 4)
  7. CLI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 1. constants
# ----------------------------------------------------------------------------
SEED = 20261003                 # the date the plan was fixed; never change
N_TRAIN = 8_000
N_DEV = 2_000
TAP_LAYERS = (12, 16, 20, 24)   # depths whose hidden states are cached (knob ② in the plan)
QTYPES = ("noul", "choice", "score")   # JevBench's names: noul = yes/no
N_SEEDS = 3
GATE_SE = 2.0                   # keep only if improvement > GATE_SE * standard error

# Corpus: Kev's decision-v7. train.jsonl (12,576 records / 15,576 questions) is on the Hub,
# not in the Kev repo, and carries no licence of its own (its ten sources include Yelp, which is
# restrictive), so this repo commits ROW IDS ONLY under data/splits/ and re-downloads rows.
HF_SUITES_REPO = "jaredpalmer/kev-suites"
HF_TRAIN_PATH = "v7/decision-v7/train.jsonl"
# Kev's record: {"state": str|json, "questions": {qid: {"type": noul|choice|score, "instructions",
#   "criteria": {key: desc} | [level, ...], "label": key|bool|int, "src"}}, "_meta": {"source", ...}}
# Families (manifest trainable_sources): agnews amazon banking77 boolq dbpedia14 imdb mnli sst5
# trec yelp legacy_policy compositional. Two are held out of training entirely (plan §6).
HELD_OUT_FAMILIES = ("trec", "legacy_policy")   # PROPOSED; confirmed in step 2 of the plan
FAMILY_KEY = ("_meta", "source")

# Sequence layout (knob "read vectors" in plan §2). Markers reuse Qwen's existing reserved tokens,
# as Kev does, so no new embedding rows have to be learned while the embedding table is frozen:
#   <state> text </state> <q> instructions <opt> option </opt> <opt> option </opt> ... <decide>
# We read the hidden state at each </opt> (one per option) and at <decide> (= the plan's <answer>).
# One causal row per question: Qwen3.5's GatedDeltaNet layers carry state across positions and
# would leak one question into the next if several were packed into a row.
QWEN_MARKERS = {
    "state": "<|fim_prefix|>", "q": "<|fim_middle|>",
    "opt": "<|box_start|>", "opt_end": "<|box_end|>", "decide": "<|fim_suffix|>",
}
MAX_STATE_TOKENS, MAX_ROW_TOKENS = 384, 2048

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"     # committed: row ids only
DATA_CACHE = ROOT / "data" / "cache"
RESULTS_TSV = ROOT / "results.tsv"


# ----------------------------------------------------------------------------
# 2. scoring
# ----------------------------------------------------------------------------
def score(probs, labels, qtypes, n_options, n_bins: int = 10) -> dict:
    """Score a set of decisions.

    probs      [N, K_max] float, row r is a distribution over its first n_options[r] entries
    labels     [N] int, index of the correct option
    qtypes     length-N sequence of strings from QTYPES
    n_options  [N] int, number of real options in each row

    Returns a dict with overall accuracy, brier (multi-class sum form, in [0, 2]),
    ece (top-label, equal-width bins), the composite `selection` score used by the
    keep rule, its analytic standard error `selection_se`, and `per_type` details.

    selection = mean over question types present of
                0.5 * ( (acc - chance) / (1 - chance)  +  (1 - brier) )
    The first term is JevBench's chance-corrected accuracy: 0 at chance, 1 when
    perfect, so a 2-option and a 6-option question count the same. The second is
    a proper scoring rule, so it rewards honest probabilities, not just the right
    argmax. Both are means of per-row quantities, which is what makes the
    standard error analytic.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_options = np.asarray(n_options, dtype=np.int64)
    qtypes = np.asarray(list(qtypes))
    N, K = probs.shape
    if not (labels.shape == (N,) and n_options.shape == (N,) and qtypes.shape == (N,)):
        raise ValueError("probs, labels, qtypes, n_options must agree on N")
    unknown = set(qtypes) - set(QTYPES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}; expected {QTYPES}")

    valid = np.arange(K)[None, :] < n_options[:, None]            # [N, K]
    p = np.where(valid, probs, 0.0)
    sums = p.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1e-3):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(f"row {bad}: probabilities over valid options sum to {sums[bad]:.4f}, not 1")
    if np.any(labels >= n_options) or np.any(labels < 0):
        raise ValueError("a label indexes a non-existent option")

    pred = np.argmax(np.where(valid, p, -np.inf), axis=1)
    correct = (pred == labels).astype(np.float64)
    chance = 1.0 / n_options
    corrected = (correct - chance) / (1.0 - chance)
    onehot = np.zeros_like(p)
    onehot[np.arange(N), labels] = 1.0
    brier = ((p - onehot) ** 2 * valid).sum(axis=1)
    conf = p.max(axis=1)
    per_row = 0.5 * (corrected + (1.0 - brier))

    per_type: dict[str, dict] = {}
    type_means, type_vars = [], []
    for t in QTYPES:
        idx = qtypes == t
        n_t = int(idx.sum())
        if n_t == 0:
            continue
        c = per_row[idx]
        per_type[t] = {
            "n": n_t,
            "accuracy": float(correct[idx].mean()),
            "chance": float(chance[idx].mean()),
            "acc_above_chance": float(corrected[idx].mean()),
            "brier": float(brier[idx].mean()),
            "selection": float(c.mean()),
        }
        type_means.append(c.mean())
        type_vars.append(c.var(ddof=1) / n_t if n_t > 1 else 0.0)
    T = len(type_means)
    selection = float(np.mean(type_means))
    selection_se = float(math.sqrt(sum(type_vars)) / T)

    return {
        "n": N,
        "accuracy": float(correct.mean()),
        "brier": float(brier.mean()),
        "ece": _ece(conf, correct, n_bins),
        "selection": selection,
        "selection_se": selection_se,
        "per_type": per_type,
    }


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Expected calibration error: bin decisions by confidence, average |accuracy - confidence|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # confidence of 1.0 falls in the last bin
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        idx = bins == b
        if idx.any():
            ece += idx.mean() * abs(correct[idx].mean() - conf[idx].mean())
    return float(ece)


def keep(baseline_selection, candidate_selection, selection_se: float) -> bool:
    """The keep rule: the mean over seeds must improve by MORE than GATE_SE standard errors."""
    delta = float(np.mean(candidate_selection)) - float(np.mean(baseline_selection))
    return delta > GATE_SE * selection_se


# ----------------------------------------------------------------------------
# 3. data
# ----------------------------------------------------------------------------
def download_corpus(dest: Path | None = None) -> Path:
    """Fetch decision-v7 train.jsonl from the Hub (public, ~19 MB). Returns the local path."""
    dest = Path(dest or (Path(__import__("os").environ.get("DECIDER_DATA", DATA_RAW.parent)) / "raw" / "train.jsonl"))
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(HF_SUITES_REPO, HF_TRAIN_PATH, repo_type="dataset")
    import shutil
    shutil.copyfile(got, dest)
    return dest


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _option_list(q: dict) -> tuple[list[tuple[str, str | None]], int]:
    """Turn a Kev question into an ordered option list and an integer label."""
    t, crit, label = q["type"], q.get("criteria"), q["label"]
    if t == "noul":
        crit = crit or {}
        options = [("no", crit.get("false")), ("yes", crit.get("true"))]
        return options, int(bool(label))
    if t == "score":
        return [(str(i), desc) for i, desc in enumerate(crit)], int(label)
    if t == "choice":
        keys = list(crit.keys())
        return [(k, crit[k]) for k in keys], keys.index(label)
    raise ValueError(f"unknown question type {t!r}")


def as_text(x) -> str:
    """Kev stores some states and instructions as JSON (dicts, lists of chat turns); render them."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and all(isinstance(v, str) for v in x.values()):
        return "\n".join(f"{k}: {v}" for k, v in x.items())
    return json.dumps(x, ensure_ascii=False)


def flatten(records: list[dict]) -> list[dict]:
    """One row per question. Kev packs several questions per record; the torso sees one per row."""
    rows = []
    for rec in records:
        meta = rec["_meta"]
        state = as_text(rec["state"])
        for qid, q in rec["questions"].items():
            options, label = _option_list(q)
            rows.append({
                "id": f"{meta['id']}#{qid}",
                "group": meta.get("group_id", meta["id"]),
                "family": meta.get("source", q.get("src")),
                "qtype": q["type"],
                "state": state,
                "instructions": as_text(q["instructions"]),
                "options": options,
                "label": label,
                "n_options": len(options),
            })
    return rows


def build_splits(rows: list[dict], seed: int = SEED, n_train: int = N_TRAIN, n_dev: int = N_DEV,
                 held_out=HELD_OUT_FAMILIES) -> dict[str, list[str]]:
    """Fix train / dev / heldout as lists of row ids.

    - heldout: every row whose family is in held_out (never trained on, reported only).
    - dev: n_dev rows stratified by question type, chosen group-by-group so that no
      group_id (Kev's paraphrase / minimal-pair grouping) is split across train and dev.
    - train: n_train rows from the remaining groups.
    Rows left over are simply unused. Deterministic in the seed.
    """
    rng = np.random.default_rng(seed)
    heldout = [r["id"] for r in rows if r["family"] in held_out]
    pool = [r for r in rows if r["family"] not in held_out]
    groups: dict[str, list[dict]] = {}
    for r in pool:
        groups.setdefault(r["group"], []).append(r)
    order = sorted(groups)
    rng.shuffle(order)

    quota = {t: n_dev // len(QTYPES) for t in QTYPES}
    for t in QTYPES[: n_dev % len(QTYPES)]:
        quota[t] += 1
    have = {t: 0 for t in QTYPES}
    dev: list[str] = []
    used: set[str] = set()
    # pass 1: whole groups that fit under every type quota
    for g in order:
        contrib = {t: sum(r["qtype"] == t for r in groups[g]) for t in QTYPES}
        if all(have[t] + contrib[t] <= quota[t] for t in QTYPES) and any(contrib.values()):
            dev += [r["id"] for r in groups[g]]
            for t in QTYPES:
                have[t] += contrib[t]
            used.add(g)
        if all(have[t] == quota[t] for t in QTYPES):
            break
    # pass 2: fill any deficit row-by-row from unused groups (the group is then burnt)
    for g in order:
        if all(have[t] == quota[t] for t in QTYPES):
            break
        if g in used:
            continue
        take = [r for r in groups[g] if have[r["qtype"]] < quota[r["qtype"]]]
        if take:
            for r in take:
                if have[r["qtype"]] < quota[r["qtype"]]:
                    dev.append(r["id"])
                    have[r["qtype"]] += 1
            used.add(g)
    if len(dev) != n_dev:
        raise ValueError(f"could only stratify {len(dev)} dev rows, wanted {n_dev}")

    train: list[str] = []
    for g in order:
        if g in used:
            continue
        train += [r["id"] for r in groups[g]]
        used.add(g)
        if len(train) >= n_train:
            break
    if len(train) < n_train:
        raise ValueError(f"only {len(train)} rows available for train, wanted {n_train}")
    return {"train": train[:n_train], "dev": dev, "heldout": heldout}


def write_splits(splits: dict[str, list[str]], raw_path: Path, rows: list[dict], out_dir: Path = DATA_SPLITS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")
    by_id = {r["id"]: r for r in rows}
    manifest = {
        "seed": SEED, "n_train": N_TRAIN, "n_dev": N_DEV, "held_out_families": list(HELD_OUT_FAMILIES),
        "source": {"repo": HF_SUITES_REPO, "path": HF_TRAIN_PATH, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
        "counts": {name: {"rows": len(ids),
                          "by_type": {t: sum(by_id[i]["qtype"] == t for i in ids) for t in QTYPES},
                          "by_family": dict(sorted(Counter(by_id[i]["family"] for i in ids).items()))}
                   for name, ids in splits.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def load_split(name: str, raw_path: Path | None = None) -> list[dict]:
    """Rows of a split in the committed id order."""
    ids = (DATA_SPLITS / f"{name}.txt").read_text().split()
    rows = flatten(load_records(download_corpus(raw_path)))
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in ids]


# ----------------------------------------------------------------------------
# 4. sequence layout
# ----------------------------------------------------------------------------
def option_text(qtype: str, key: str, desc: str | None) -> str:
    if qtype == "score" and desc:
        return desc
    return f"{key}: {desc}" if desc else key


def encode_row(tok, row: dict, markers: dict = QWEN_MARKERS, shuffle: bool = False, rng=None) -> dict:
    """Tokenise one row as  <state> s </state>?  — see the layout note next to QWEN_MARKERS.

        [state] s_1..s_n [q] i_1..i_m ([opt] o_1..o_k [opt_end])*N [decide]

    Returns ids plus the positions the head reads: decide_pos (the plan's <answer>) and
    opt_pos (one per option, at its [opt_end]). With shuffle=True a choice question's
    options are permuted and the label moves with them; noul and score keep their order
    because the order carries meaning (no/yes, increasing level).
    """
    m = {k: tok.convert_tokens_to_ids(v) for k, v in markers.items()}
    n = row["n_options"]
    perm = list(range(n))
    if shuffle and row["qtype"] == "choice":
        rng = rng or np.random.default_rng()
        perm = [int(i) for i in rng.permutation(n)]
    label = perm.index(row["label"])

    state_ids = tok.encode(row["state"], add_special_tokens=False)[:MAX_STATE_TOKENS]
    instr_ids = tok.encode(row["instructions"], add_special_tokens=False)
    opt_ids = [tok.encode(option_text(row["qtype"], *row["options"][j]), add_special_tokens=False) for j in perm]

    fixed = len(state_ids) + len(instr_ids) + 3 + 2 * n
    budget = MAX_ROW_TOKENS - fixed
    if sum(map(len, opt_ids)) > budget:
        cap = max(1, budget // n)
        opt_ids = [o[:cap] for o in opt_ids]

    ids = [m["state"], *state_ids, m["q"], *instr_ids]
    opt_pos = []
    for o in opt_ids:
        ids += [m["opt"], *o, m["opt_end"]]
        opt_pos.append(len(ids) - 1)
    ids.append(m["decide"])
    return {
        "ids": ids, "decide_pos": len(ids) - 1, "opt_pos": opt_pos, "label": label,
        "qtype": row["qtype"], "n_options": n, "perm": perm, "state_tokens": len(state_ids), "id": row["id"],
    }


# ----------------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the corpus and write the fixed split ids (step 2)")
    sub.add_parser("cache", help="cache read-out hidden states for TAP_LAYERS (step 4)")
    sub.add_parser("tier-a", help="train head.py on the cache with N_SEEDS seeds and append results.tsv")
    args = ap.parse_args(argv)
    if args.cmd == "data":
        raw = download_corpus()
        rows = flatten(load_records(raw))
        splits = build_splits(rows)
        write_splits(splits, raw, rows)
        print(json.dumps(json.loads((DATA_SPLITS / "manifest.json").read_text())["counts"], indent=1))
        return
    raise SystemExit(f"'{args.cmd}' is implemented in a later step of the plan; see program.md")


if __name__ == "__main__":
    main()
