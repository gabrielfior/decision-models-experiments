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
import math
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
# 7. CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the corpus and write the fixed split ids (step 2)")
    sub.add_parser("cache", help="cache read-out hidden states for TAP_LAYERS (step 4)")
    sub.add_parser("tier-a", help="train head.py on the cache with N_SEEDS seeds and append results.tsv")
    args = ap.parse_args(argv)
    raise SystemExit(f"'{args.cmd}' is implemented in a later step of the plan; see program.md")


if __name__ == "__main__":
    main()
