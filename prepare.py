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
MAX_STATE_TOKENS, MAX_INSTR_TOKENS, MAX_ROW_TOKENS = 384, 512, 2048

# Hybrid read-out (head "hybrid"): each presented option is prefixed with its slot letter, "A) ", "B) ", ...
# in PRESENTED order (after shuffling the letter follows the position, not the option), so the torso's own
# next-token logit for " A", " B", ... at <decide> — h_last · E[letter] with tied embeddings — can be read
# alongside the pointer score. Default OFF; train.py / serve.py switch it on when the head is "hybrid".
LETTER_PREFIX = False
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Option preview ("option repetition", arXiv 2601.14152): a compact list of the options is placed right after
# the [state] marker, BEFORE the state text, so the options are visible while the state is read:
#   [state] Options: o_1 | o_2 | ... s_1..s_n [q] ...
# The preview is in PRESENTED order (the same permutation as the full [opt] list), each option cut to
# OPTION_PREVIEW_TOKENS tokens; pieces are tokenised separately and concatenated at the token level. The preview
# counts toward MAX_ROW_TOKENS; when a row overflows, the state is truncated first (never below
# OPTION_PREVIEW_MIN_STATE tokens), then the options are capped as usual. Default OFF; train.py --option-preview
# turns it on and records it in config.json, which serve.py honours.
OPTION_PREVIEW = False
OPTION_PREVIEW_PREFIX, OPTION_PREVIEW_SEP = "Options:", " |"
OPTION_PREVIEW_TOKENS = 6
OPTION_PREVIEW_MIN_STATE = 64

# The torso ladder (plan §3). Each torso brings its own single-token markers (chosen from tokens
# the tokenizer already has) and the depths tapped for the read-out, at 50/67/83/100% of depth.
TORSOS = {
    "Qwen/Qwen3.5-0.8B-Base": {"markers": QWEN_MARKERS, "tap_layers": TAP_LAYERS, "n_layers": 24,
                               # Kev's targets: attention, MLP and the GatedDeltaNet projections
                               "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]},
    # Scaling check only (plan §3): same family and layout, 24 layers at d = 2048 (18 linear + 6 full attention).
    "Qwen/Qwen3.5-2B-Base": {"markers": QWEN_MARKERS, "tap_layers": TAP_LAYERS, "n_layers": 24,
                             "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                              "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]},
    # LFM2.5: 14 layers (8 short-conv + 6 attention), all causal. FIM tokens saw pretraining;
    # the tool-list tokens are single added tokens with meaningful "list of items" semantics.
    "LiquidAI/LFM2.5-230M-Base": {"markers": {"state": "<|fim_pre|>", "q": "<|fim_mid|>", "opt": "<|tool_list_start|>",
                                               "opt_end": "<|tool_list_end|>", "decide": "<|fim_suf|>"},
                                  "tap_layers": (7, 9, 12, 14), "n_layers": 14,
                                  # attention q/k/v/out, short-conv in/out ("out_proj" matches both), MLP w1/w2/w3
                                  "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]},
    # ModernBERT: 28 bidirectional layers. [SEP] only ever closed a whole sequence in pretraining,
    # so dedicated [unusedN] tokens mark the parts; their embeddings are untrained, so the LoRA/
    # full fine-tune of Tier B has to learn them (Tier A on a frozen torso will be weak here).
    "answerdotai/ModernBERT-large": {"markers": {"state": "[unused0]", "q": "[unused1]", "opt": "[unused2]",
                                                 "opt_end": "[unused3]", "decide": "[unused4]"},
                                     "tap_layers": (14, 19, 23, 28), "n_layers": 28, "bidirectional": True,
                                     # fused qkv + output projection in attention, Wi (GeGLU in) and Wo (shared name) in the MLP
                                     "lora_targets": ["Wqkv", "Wo", "Wi"]},
}


def torso_config(name: str) -> dict:
    """Markers and tap layers for a torso, by full Hub id or by its short name."""
    if name in TORSOS:
        return TORSOS[name]
    for k, v in TORSOS.items():
        if k.split("/")[-1] == name.split("/")[-1]:
            return v
    raise KeyError(f"unknown torso {name!r}; add it to prepare.TORSOS")

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"     # committed: row ids only
DATA_CACHE = ROOT / "data" / "cache"
RESULTS_TSV = Path(__import__("os").environ.get("DECIDER_RESULTS", ROOT / "results.tsv"))
EDITABLE_FILES = [ROOT / "head.py", ROOT / "train.py"]   # hashed into every results row for provenance


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


def noise_floor(baseline_selection, candidate_selection, selection_se: float) -> float:
    """The standard error the gate measures against: the larger of the dev-set SE (how much the
    score would move if we drew another 2k dev rows) and the seed SE (how much it moves if we
    retrain the same head with another seed). Both are noise; the gate must clear both."""
    seed_ses = [float(np.std(x, ddof=1)) / math.sqrt(len(x)) for x in (baseline_selection, candidate_selection) if len(x) > 1]
    return max([float(selection_se), *seed_ses])


def keep(baseline_selection, candidate_selection, selection_se: float) -> bool:
    """The keep rule: the mean over seeds must improve by MORE than GATE_SE noise floors."""
    delta = float(np.mean(candidate_selection)) - float(np.mean(baseline_selection))
    return delta > GATE_SE * noise_floor(baseline_selection, candidate_selection, selection_se)


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


def _option_list(q: dict) -> tuple[list[tuple[str, str | None]], int | None]:
    """Turn a Kev/JevBench question into an ordered option list and an integer label (None if unlabeled)."""
    t, crit, label = q["type"], q.get("criteria"), q.get("label")
    if t == "noul":
        crit = crit or {}
        options = [("no", crit.get("false")), ("yes", crit.get("true"))]
        return options, (None if label is None else int(bool(label)))
    if t == "score":
        return [(str(i), desc) for i, desc in enumerate(crit)], (None if label is None else int(label))
    if t == "choice":
        keys = list(crit.keys())
        return [(k, crit[k]) for k in keys], (None if label is None else keys.index(label))
    raise ValueError(f"unknown question type {t!r}")


def question_to_row(state, q: dict, qid: str = "decision", rid: str = "live") -> dict:
    """A row for an unlabeled question at serving time (same layout as flatten())."""
    options, label = _option_list(q)
    return {"id": f"{rid}#{qid}", "group": rid, "family": "live", "qtype": q["type"], "state": as_text(state),
            "instructions": as_text(q["instructions"]), "options": options, "label": label, "n_options": len(options)}


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


def letter_token_ids(tok) -> list[int]:
    """Token id of " A", " B", ... " Z" (one per LETTERS entry). Qwen's tokenizer has each as a single
    token; a tokenizer that splits one falls back to that letter's first id."""
    ids = []
    for c in LETTERS:
        enc = tok.encode(" " + c, add_special_tokens=False)
        ids.append(int(enc[0]))              # single token for Qwen; else the first piece
    return ids


def letter_embeddings(model, tok):
    """The torso's input-embedding rows for the letter tokens, [26, d], detached (the frozen table; with tied
    embeddings these rows are also the LM-head rows, so h_last · E[letter] is the next-token logit)."""
    import torch
    with torch.no_grad():
        return model.get_input_embeddings().weight[torch.as_tensor(letter_token_ids(tok))].detach().clone()


def option_preview_ids(tok, qtype: str, options: list[tuple[str, str | None]], perm: list[int]) -> list[int]:
    """Token ids of the option preview in PRESENTED order: OPTION_PREVIEW_PREFIX, then option_text() of
    options[perm[j]] cut to OPTION_PREVIEW_TOKENS tokens (the raw key when the description is None, as
    option_text already does), joined by OPTION_PREVIEW_SEP. Each piece is tokenised on its own."""
    ids = tok.encode(OPTION_PREVIEW_PREFIX, add_special_tokens=False)
    sep = tok.encode(OPTION_PREVIEW_SEP, add_special_tokens=False)
    for slot, j in enumerate(perm):
        if slot:
            ids = ids + sep
        ids = ids + tok.encode(option_text(qtype, *options[j]), add_special_tokens=False)[:OPTION_PREVIEW_TOKENS]
    return ids


def encode_row(tok, row: dict, markers: dict = QWEN_MARKERS, shuffle: bool = False, rng=None,
               order: str = "identity", letter_prefix: bool | None = None, option_preview: bool | None = None) -> dict:
    """Tokenise one row as  <state> s </state>?  — see the layout note next to QWEN_MARKERS.

        [state] (preview)? s_1..s_n [q] i_1..i_m ([opt] o_1..o_k [opt_end])*N [decide]

    Returns ids plus the positions the head reads: decide_pos (the plan's <answer>) and
    opt_pos (one per option, at its [opt_end]). With shuffle=True a choice question's
    options are permuted and the label moves with them; noul and score keep their order
    because the order carries meaning (no/yes, increasing level).
    letter_prefix (None = module LETTER_PREFIX) prefixes the option shown in slot j with
    f"{LETTERS[j]}) " for j < 26, for the hybrid letter-logit read-out.
    option_preview (None = module OPTION_PREVIEW) inserts option_preview_ids() under the SAME
    permutation between the [state] marker and the state text; see the note next to OPTION_PREVIEW.
    """
    if letter_prefix is None:
        letter_prefix = LETTER_PREFIX
    if option_preview is None:
        option_preview = OPTION_PREVIEW
    m = {k: tok.convert_tokens_to_ids(v) for k, v in markers.items()}
    n = row["n_options"]
    perm = list(range(n))
    if row["qtype"] == "choice":
        if shuffle:
            rng = rng or np.random.default_rng()
            perm = [int(i) for i in rng.permutation(n)]
        elif order == "reversed":
            perm = perm[::-1]
        elif order.startswith("shift:"):                 # cyclic rotation by k: presented position j shows option (j + k) mod n
            k = int(order.split(":")[1]) % n
            perm = perm[k:] + perm[:k]
    label = None if row["label"] is None else perm.index(row["label"])

    state_ids = tok.encode(row["state"], add_special_tokens=False)[:MAX_STATE_TOKENS]
    instr_ids = tok.encode(row["instructions"], add_special_tokens=False)[:MAX_INSTR_TOKENS]
    def _shown(slot: int, j: int) -> str:
        text = option_text(row["qtype"], *row["options"][j])
        return f"{LETTERS[slot]}) {text}" if letter_prefix and slot < len(LETTERS) else text
    opt_ids = [tok.encode(_shown(slot, j), add_special_tokens=False) for slot, j in enumerate(perm)]
    preview_ids = option_preview_ids(tok, row["qtype"], row["options"], perm) if option_preview else []

    fixed = len(state_ids) + len(instr_ids) + len(preview_ids) + 3 + 2 * n
    budget = MAX_ROW_TOKENS - fixed
    if option_preview and sum(map(len, opt_ids)) > budget:
        # the preview took room from the options: give the state up first (down to a floor), then cap options
        keep_state = max(OPTION_PREVIEW_MIN_STATE, len(state_ids) - (sum(map(len, opt_ids)) - budget))
        if keep_state < len(state_ids):
            state_ids = state_ids[:keep_state]
            fixed = len(state_ids) + len(instr_ids) + len(preview_ids) + 3 + 2 * n
            budget = MAX_ROW_TOKENS - fixed
    if sum(map(len, opt_ids)) > budget:
        cap = max(1, budget // n)
        opt_ids = [o[:cap] for o in opt_ids]

    ids = [m["state"], *preview_ids, *state_ids, m["q"], *instr_ids]
    opt_pos = []
    for o in opt_ids:
        ids += [m["opt"], *o, m["opt_end"]]
        opt_pos.append(len(ids) - 1)
    ids.append(m["decide"])
    return {
        "ids": ids, "decide_pos": len(ids) - 1, "opt_pos": opt_pos, "label": label,
        "qtype": row["qtype"], "n_options": n, "perm": perm, "state_tokens": len(state_ids),
        "instr_tokens": len(instr_ids), "id": row["id"],
    }


# ----------------------------------------------------------------------------
# 5. torso, batching, prediction, calibration, reporting
# ----------------------------------------------------------------------------
def load_torso(name: str, dtype=None, device: str = "cpu"):
    """Load a torso WITHOUT its language-model head: AutoModel gives the bare stack of layers.

    Qwen3.5 base checkpoints are multimodal; we keep only the text stack (`language_model`).
    Returns (tokenizer, model, d_model).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=dtype or torch.float32)
    model = getattr(model, "language_model", model)
    model.to(device)
    cfg = getattr(model.config, "text_config", model.config)
    return tok, model, int(cfg.hidden_size)


def batches(tok, rows: list[dict], batch_size: int, shuffle_options: bool, rng=None, device: str = "cpu",
            order: str = "identity", shuffle_rows: bool = False, markers: dict = QWEN_MARKERS,
            letter_prefix: bool | None = None, option_preview: bool | None = None):
    """Encode rows and yield right-padded tensor batches with the read-out positions.
    letter_prefix and option_preview are passed through to encode_row (None = module LETTER_PREFIX / OPTION_PREVIEW)."""
    import torch
    idx = list(range(len(rows)))
    if shuffle_rows:
        (rng or np.random.default_rng()).shuffle(idx)
    pad_id = getattr(tok, "pad_token_id", None) or 0
    for start in range(0, len(idx), batch_size):
        encs = [encode_row(tok, rows[i], markers, shuffle=shuffle_options, rng=rng, order=order, letter_prefix=letter_prefix,
                           option_preview=option_preview)
                for i in idx[start:start + batch_size]]
        B, T, K = len(encs), max(len(e["ids"]) for e in encs), max(e["n_options"] for e in encs)
        ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        opt_pos = torch.full((B, K), -1, dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
            attn[i, : len(e["ids"])] = 1
            opt_pos[i, : e["n_options"]] = torch.tensor(e["opt_pos"])
        yield {
            "ids": ids.to(device), "attn": attn.to(device),
            "decide_pos": torch.tensor([e["decide_pos"] for e in encs], device=device),
            "opt_pos": opt_pos.to(device), "opt_mask": (opt_pos >= 0).to(device),
            "label": torch.tensor([-1 if e["label"] is None else e["label"] for e in encs], device=device),
            "qtype": [e["qtype"] for e in encs], "n_options": [e["n_options"] for e in encs],
            "perm": [e["perm"] for e in encs], "id": [e["id"] for e in encs],
        }


def predict_rows(logit_fn, tok, rows: list[dict], order: str = "identity", device: str = "cpu", batch_size: int = 16,
                 markers: dict = QWEN_MARKERS) -> list[dict]:
    """Run `logit_fn(batch) -> [B, K]` over rows; return logits mapped back to CANONICAL option order.

    Mapping back lets two runs under different option orders be compared row by row.
    Each prediction also carries the wall-clock cost of its batch, split per row, in ms.
    """
    import time
    import torch
    preds = []
    with torch.no_grad():
        for b in batches(tok, rows, batch_size, shuffle_options=False, order=order, device=device, markers=markers):
            t0 = time.perf_counter()
            z = logit_fn(b).float().cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000 / len(b["id"])
            for i, (perm, n) in enumerate(zip(b["perm"], b["n_options"])):
                canon = np.empty(n)
                canon[perm] = z[i, :n]                     # canon[perm[j]] = z_j
                preds.append({"logits": canon, "label": int(perm[int(b["label"][i])]), "qtype": b["qtype"][i],
                              "n_options": n, "id": b["id"][i], "ms": ms})
    return preds


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def probs_from(preds: list[dict], temps: dict[str, float]) -> list[np.ndarray]:
    """Boltzmann distribution over options, p_i ∝ exp(z_i / T) with T per question type."""
    return [_softmax(np.asarray(p["logits"]) / temps.get(p["qtype"], 1.0)) for p in preds]


def fit_temperature_by_type(preds: list[dict], log_t_range=(math.log(0.05), math.log(20.0)), n_grid: int = 400) -> dict[str, float]:
    """Per question type, the T that minimises log loss on these predictions. One scalar each."""
    temps = {}
    grid = np.exp(np.linspace(*log_t_range, n_grid))
    for t in QTYPES:
        sub = [p for p in preds if p["qtype"] == t]
        if not sub:
            continue
        best_T, best_nll = 1.0, float("inf")
        for T in grid:
            nll = 0.0
            for p in sub:
                pr = _softmax(np.asarray(p["logits"]) / T)
                nll -= math.log(max(pr[p["label"]], 1e-12))
            if nll < best_nll:
                best_T, best_nll = float(T), nll
        temps[t] = best_T
    return temps


def score_preds(preds: list[dict], temps: dict[str, float]) -> dict:
    """Pad per-row probability vectors into the [N, K_max] matrix score() wants."""
    if not preds:
        return {}
    probs = probs_from(preds, temps)
    K = max(p["n_options"] for p in preds)
    mat = np.zeros((len(preds), K))
    for i, pr in enumerate(probs):
        mat[i, : len(pr)] = pr
    return score(mat, [p["label"] for p in preds], [p["qtype"] for p in preds], [p["n_options"] for p in preds])


def evaluate(dev_a: list[dict], dev_b: list[dict], heldout: list[dict], temps: dict[str, float]) -> dict:
    """The report every experiment produces: dev under order A (selection), order sensitivity
    between orders A and B, held-out families (reported only), and latency."""
    dev = score_preds(dev_a, temps)
    held = score_preds(heldout, temps)
    flips = [int(np.argmax(a["logits"]) != np.argmax(b["logits"])) for a, b in zip(dev_a, dev_b)]
    ms = [p["ms"] for p in dev_a if not math.isnan(p.get("ms", float("nan")))]
    return {
        "dev_selection": dev["selection"], "dev_se": dev["selection_se"], "dev_acc": dev["accuracy"],
        "dev_brier": dev["brier"], "dev_ece": dev["ece"],
        "heldout_acc": held.get("accuracy", float("nan")), "heldout_brier": held.get("brier", float("nan")),
        "order_sens": float(np.mean(flips)) if flips else float("nan"),
        "p50_ms": float(np.nanmedian(ms)) if ms else float("nan"),
        "temps": temps, "dev": dev, "heldout": held,
    }


RESULT_COLUMNS = ["timestamp", "commit", "tier", "note", "seeds", "dev_selection", "dev_se", "dev_acc", "dev_brier",
                  "dev_ece", "heldout_acc", "heldout_brier", "order_sens", "p50_ms", "kept", "per_seed", "baseline", "code_hash"]


def git_commit() -> str:
    """Short HEAD sha, with a -dirty suffix when the working tree differs from it (the usual case
    while an experiment is being run before its commit)."""
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "head.py", "train.py"], cwd=ROOT, stderr=subprocess.DEVNULL).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def code_hash() -> str:
    """8 hex chars identifying the exact contents of the editable files that produced a row."""
    h = hashlib.sha1()
    for f in EDITABLE_FILES:
        h.update(Path(f).read_bytes() if Path(f).exists() else b"")
    return h.hexdigest()[:8]


def append_results(rep: dict, path: Path = RESULTS_TSV, commit: str | None = None) -> None:
    import time
    path = Path(path)
    import os
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit or os.environ.get("DECIDER_COMMIT") or git_commit(),
           "code_hash": code_hash(), **rep}
    header = not path.exists() or path.stat().st_size == 0
    with open(path, "a") as f:
        if header:
            f.write("\t".join(RESULT_COLUMNS) + "\n")
        f.write("\t".join(_fmt(row.get(c, "")) for c in RESULT_COLUMNS) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return ",".join(_fmt(x) for x in v)
    if v is None:
        return ""
    return str(v).replace("\t", " ").replace("\n", " ")


def read_baseline(path: Path = RESULTS_TSV, tier: str = "A", match: str | None = None) -> list[float] | None:
    """Per-seed dev selection scores of the most recent KEPT experiment of this tier, or None.
    With `match`, the most recent kept row whose note contains that substring (a pinned baseline,
    so a lane of variants is always compared with the same reference instead of its own last keep).
    Falls back to the single mean when the row predates per-seed logging."""
    path = Path(path)
    if not path.exists():
        return None
    import csv
    kept = [r for r in csv.DictReader(open(path), delimiter="\t") if r["tier"] == tier and r["kept"] == "True"
            and (match is None or match in r["note"])]
    if not kept:
        return None
    per_seed = kept[-1].get("per_seed", "")
    return [float(x) for x in per_seed.split(",")] if per_seed else [float(kept[-1]["dev_selection"])]


# ----------------------------------------------------------------------------
# 5b. feature cache
# ----------------------------------------------------------------------------
def cache_dir() -> Path:
    import os
    return Path(os.environ.get("DECIDER_CACHE", DATA_CACHE))


def cache_features(torso_name: str, splits=("train", "dev", "heldout"), batch_size: int = 16, device: str | None = None,
                   out_dir: Path | None = None) -> None:
    """Run the frozen torso once over every split and save the read-out vectors at TAP_LAYERS.

    Saved per split as .npz, float16, option vectors ragged with `offsets`:
        h_ans   [N, L, d]          state at <decide>
        h_opts  [sum_i n_i, L, d]  state at each </opt>, rows of question i at offsets[i]:offsets[i+1]
        offsets, labels, qtypes, n_options, ids, d_model
        letter_emb  [26, d]        input-embedding rows of " A".." Z" (hybrid head; the deepest tap of h_ans
                                   is the last layer's post-norm state at <decide>, so h_ans[:, -1] · letter_emb
                                   is the torso's own letter logit)
    Option order is the canonical one. This is the measurement step of the plan: a few
    observables taken from a large state, after which the torso leaves the loop.
    """
    import time
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir or cache_dir() / torso_name.split("/")[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = torso_config(torso_name)
    markers, taps = cfg["markers"], cfg["tap_layers"]
    tok, model, d = load_torso(torso_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device)
    model.eval()
    letter_emb = letter_embeddings(model, tok).to(torch.float16).cpu().numpy()
    for split in splits:
        rows = load_split(split)
        h_ans, h_opts, labels, qtypes, n_opts, ids = [], [], [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for bi, b in enumerate(batches(tok, rows, batch_size, shuffle_options=False, device=device, markers=markers)):
                out = model(input_ids=b["ids"], attention_mask=b["attn"], output_hidden_states=True)
                hs = torch.stack([out.hidden_states[l] for l in taps], dim=1)            # [B, L, T, d]
                ar = torch.arange(hs.shape[0], device=device)
                h_ans.append(hs[ar, :, b["decide_pos"]].to(torch.float16).cpu().numpy())
                for i, n in enumerate(b["n_options"]):
                    h_opts.append(hs[i][:, b["opt_pos"][i, :n]].permute(1, 0, 2).to(torch.float16).cpu().numpy())
                labels += b["label"].tolist(); qtypes += b["qtype"]; n_opts += b["n_options"]; ids += b["id"]
                if bi % 50 == 0:
                    print(f"{split}: {len(ids)}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        n_opts = np.array(n_opts)
        np.savez(out_dir / f"{split}.npz", h_ans=np.concatenate(h_ans), h_opts=np.concatenate(h_opts),
                 offsets=np.concatenate([[0], np.cumsum(n_opts)]), labels=np.array(labels), qtypes=np.array(qtypes),
                 n_options=n_opts, ids=np.array(ids), d_model=d, tap_layers=np.array(taps), letter_emb=letter_emb)
        print(f"{split}: wrote {len(ids)} rows in {time.time() - t0:.0f}s -> {out_dir / (split + '.npz')}", flush=True)


# ----------------------------------------------------------------------------
# 6. Tier A trainer
# ----------------------------------------------------------------------------
def _load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _cache_batches(c, idx, rng, device, shuffle_options: bool, order: str = "identity"):
    """Yield padded (h_ans [B,L,d], h_opts [B,L,K,d], mask, label) from a ragged cache.

    Rows are bucketed by option count so a 77-option row never pads a 2-option batch.
    With shuffle_options the option VECTORS of choice rows are permuted along with the
    label: an augmentation that stops a head from reading the label's position. (The true
    order sensitivity, where the torso re-reads the options, is measured in Tier B.)
    """
    import torch
    L, d = c["h_ans"].shape[1:]
    for b_idx in idx:
        K = int(c["n_options"][b_idx].max())
        B = len(b_idx)
        ha = torch.from_numpy(c["h_ans"][b_idx].astype(np.float32)).to(device)
        ho = torch.zeros((B, L, K, d), device=device)
        mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        lab = torch.zeros(B, dtype=torch.long, device=device)
        for j, i in enumerate(b_idx):
            n = int(c["n_options"][i]); o0 = int(c["offsets"][i])
            perm = np.arange(n)
            if c["qtypes"][i] == "choice":
                if shuffle_options:
                    perm = rng.permutation(n)
                elif order == "reversed":
                    perm = perm[::-1]
            vecs = c["h_opts"][o0:o0 + n].astype(np.float32)[perm]          # [n, L, d]
            ho[j, :, :n] = torch.from_numpy(vecs).permute(1, 0, 2).to(device)
            mask[j, :n] = True
            lab[j] = int(np.where(perm == c["labels"][i])[0][0])
        yield ha, ho, mask, lab, b_idx


def _bucketed_indices(n_options: np.ndarray, batch_size: int, rng, shuffle: bool) -> list[np.ndarray]:
    order = np.argsort(n_options, kind="stable")
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def tier_a(head_mod, cache_dir: Path | None = None, torso: str = "Qwen3.5-0.8B-Base", seeds=tuple(range(N_SEEDS)),
           epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.01, note: str = "",
           results_path: Path = RESULTS_TSV, baseline="auto", device: str = "cpu", baseline_match: str | None = None,
           save_dir: Path | None = None) -> dict:
    """Train head_mod.build_head on the cached read-out vectors, one run per seed, and gate it.

    Returns the aggregated report (means over seeds) with `kept` decided by keep() against the
    most recent kept Tier A result in results.tsv (baseline="auto"), a number, or None (= keep).
    """
    import torch
    cdir = Path(cache_dir or (globals()["cache_dir"]() / torso))
    train, dev, held = (_load_cache(cdir / f"{s}.npz") for s in ("train", "dev", "heldout"))
    d, L = int(train["d_model"]), train["h_ans"].shape[1]
    # The hybrid head reads the torso's letter embeddings, which only the cache (not HEAD_CONFIG) carries.
    build_cfg = None
    if head_mod.HEAD_CONFIG.get("name") == "hybrid":
        if "letter_emb" not in train:
            raise ValueError(f"head 'hybrid' needs letter_emb in the cache; re-run `prepare.py cache` ({cdir / 'train.npz'} has none)")
        build_cfg = {"letter_emb": torch.from_numpy(train["letter_emb"].astype(np.float32))}
    reports = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = head_mod.build_head(d, L, build_cfg).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            head.train()
            for ha, ho, mask, lab, _ in _cache_batches(train, _bucketed_indices(train["n_options"], batch_size, rng, True), rng, device, True):
                loss = head.loss(head(ha, ho, mask), lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()

        def predict(c, order):
            preds = [None] * len(c["labels"])
            with torch.no_grad():
                for ha, ho, mask, lab, b_idx in _cache_batches(c, _bucketed_indices(c["n_options"], 256, rng, False), rng, device, False, order):
                    z = head(ha, ho, mask).float().cpu().numpy()
                    for j, i in enumerate(b_idx):
                        n = int(c["n_options"][i]); zi = z[j, :n]
                        if order == "reversed" and c["qtypes"][i] == "choice":
                            zi = zi[::-1]
                        preds[i] = {"logits": zi.copy(), "label": int(c["labels"][i]), "qtype": str(c["qtypes"][i]), "n_options": n, "id": str(c["ids"][i])}
            return preds

        dev_a, dev_b, held_p = predict(dev, "identity"), predict(dev, "reversed"), predict(held, "identity")
        temps = fit_temperature_by_type(dev_a)
        reports.append(evaluate(dev_a, dev_b, held_p, temps))
        reports[-1]["seed"], reports[-1]["state_dict"] = seed, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    agg = {k: float(np.mean([r[k] for r in reports])) for k in ("dev_selection", "dev_se", "dev_acc", "dev_brier", "dev_ece", "heldout_acc", "heldout_brier", "order_sens")}
    agg.update(p50_ms=float("nan"), tier="A", note=note, seeds=len(reports), per_seed=[r["dev_selection"] for r in reports],
               temps=reports[-1]["temps"], head=dict(head_mod.HEAD_CONFIG))
    base = read_baseline(results_path, match=baseline_match) if baseline == "auto" else baseline
    if isinstance(base, (int, float)):
        base = [float(base)]
    agg["baseline"] = base
    if base is not None:
        agg["kept"] = keep(base, agg["per_seed"], agg["dev_se"])
    else:
        # Ungated run: it anchors the file only if nothing kept exists yet; otherwise it is recorded
        # as "ungated" so read_baseline() never adopts an exploratory row as the reference.
        agg["kept"] = True if read_baseline(results_path) is None else "ungated"
    agg["noise_floor"] = None if base is None else noise_floor(base, agg["per_seed"], agg["dev_se"])
    append_results(agg, path=results_path)
    if save_dir is not None:
        import torch
        best = max(reports, key=lambda r: r["dev_selection"])
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best["state_dict"], save_dir / "head.pt")
        (save_dir / "head.json").write_text(json.dumps({"head": dict(head_mod.HEAD_CONFIG), "seed": best["seed"], "temps": best["temps"],
                                                        "torso": torso, "note": note, "dev_selection": best["dev_selection"]}, indent=1))
    verdict = "KEEP" if agg["kept"] else "DISCARD"
    print(f"{verdict}: dev selection {agg['dev_selection']:.4f} ± {agg['dev_se']:.4f} (per seed {['%.4f' % s for s in agg['per_seed']]})"
          f" vs baseline {base} (noise floor {agg['noise_floor']}); acc {agg['dev_acc']:.3f} brier {agg['dev_brier']:.3f} ece {agg['dev_ece']:.3f}"
          f" heldout acc {agg['heldout_acc']:.3f} order_sens {agg['order_sens']:.3f}")
    return agg


# ----------------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the corpus and write the fixed split ids (step 2)")
    c = sub.add_parser("cache", help="cache read-out hidden states for TAP_LAYERS (step 4)")
    c.add_argument("--torso", default="Qwen/Qwen3.5-0.8B-Base")
    c.add_argument("--batch-size", type=int, default=16)
    c.add_argument("--splits", default="train,dev,heldout")
    a = sub.add_parser("tier-a", help="train head.py on the cache with N_SEEDS seeds and append results.tsv")
    a.add_argument("--note", required=True)
    a.add_argument("--torso", default="Qwen3.5-0.8B-Base")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--no-gate", action="store_true", help="record without comparing to a baseline")
    a.add_argument("--seeds", default=",".join(str(i) for i in range(N_SEEDS)), help="comma-separated seeds, e.g. 3,4,5 for a replication")
    a.add_argument("--baseline", default=None, help="pin the gate to the latest KEPT row whose note contains this substring")
    a.add_argument("--save", default=None, help="directory to save the best seed's head.pt + head.json (for serve.py --head)")
    args = ap.parse_args(argv)
    if args.cmd == "cache":
        cache_features(args.torso, splits=tuple(args.splits.split(",")), batch_size=args.batch_size)
        return
    if args.cmd == "tier-a":
        import head as head_mod
        tier_a(head_mod, torso=args.torso, epochs=args.epochs, note=args.note, baseline=None if args.no_gate else "auto",
               seeds=tuple(int(x) for x in args.seeds.split(",")), baseline_match=args.baseline,
               save_dir=Path(args.save) if args.save else None)
        return
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
