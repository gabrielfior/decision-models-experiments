"""Tests for the frozen data section of prepare.py: flatten, split, encode."""
import json

import numpy as np
import pytest

import prepare as P


def _record(rid, source, group=None, state="some text"):
    return {
        "state": state,
        "questions": {
            "q_choice": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": None, "c": "C"}, "label": "b", "src": source},
            "q_noul": {"type": "noul", "instructions": "Yes?", "criteria": {"true": "pos", "false": "neg"}, "label": True, "src": source},
            "q_score": {"type": "score", "instructions": "Stars?", "criteria": ["1 star", "2 stars", "3 stars"], "label": 2, "src": source},
        },
        "_meta": {"id": rid, "group_id": group or rid, "source": source},
    }


# ---- flatten ------------------------------------------------------------------------
def test_flatten_makes_one_row_per_question_with_indexed_labels():
    rows = P.flatten([_record("r1", "yelp")])
    assert [r["qtype"] for r in rows] == ["choice", "noul", "score"]
    choice, noul, score = rows
    assert choice["options"] == [("a", "A"), ("b", None), ("c", "C")] and choice["label"] == 1
    assert noul["options"] == [("no", "neg"), ("yes", "pos")] and noul["label"] == 1
    assert score["options"] == [("0", "1 star"), ("1", "2 stars"), ("2", "3 stars")] and score["label"] == 2
    assert choice["id"] == "r1#q_choice" and choice["family"] == "yelp" and choice["group"] == "r1"


def test_flatten_noul_without_criteria_still_has_two_options():
    rec = _record("r1", "boolq")
    del rec["questions"]["q_noul"]["criteria"]
    rec["questions"]["q_noul"]["label"] = False
    noul = [r for r in P.flatten([rec]) if r["qtype"] == "noul"][0]
    assert noul["options"] == [("no", None), ("yes", None)] and noul["label"] == 0


# ---- split --------------------------------------------------------------------------
def _corpus():
    recs = []
    for i in range(400):
        fam = ["agnews", "yelp", "trec", "legacy_policy"][i % 4]
        recs.append(_record(f"r{i}", fam, group=f"g{i // 2}"))   # pairs share a group
    return P.flatten(recs)


def test_split_sizes_and_held_out_families_are_isolated():
    rows = _corpus()
    s = P.build_splits(rows, seed=1, n_train=300, n_dev=120, held_out=("trec", "legacy_policy"))
    assert len(s["train"]) == 300 and len(s["dev"]) == 120
    by_id = {r["id"]: r for r in rows}
    for name in ("train", "dev"):
        assert all(by_id[i]["family"] not in ("trec", "legacy_policy") for i in s[name])
    assert s["heldout"] and all(by_id[i]["family"] in ("trec", "legacy_policy") for i in s["heldout"])
    assert set(s["train"]).isdisjoint(s["dev"])


def test_split_never_puts_one_group_in_both_train_and_dev():
    rows = _corpus()
    s = P.build_splits(rows, seed=1, n_train=300, n_dev=120, held_out=("trec", "legacy_policy"))
    by_id = {r["id"]: r for r in rows}
    assert {by_id[i]["group"] for i in s["train"]}.isdisjoint({by_id[i]["group"] for i in s["dev"]})


def test_split_dev_is_stratified_by_question_type():
    rows = _corpus()
    s = P.build_splits(rows, seed=1, n_train=300, n_dev=120, held_out=("trec", "legacy_policy"))
    by_id = {r["id"]: r for r in rows}
    counts = {t: sum(by_id[i]["qtype"] == t for i in s["dev"]) for t in P.QTYPES}
    assert counts == {"noul": 40, "choice": 40, "score": 40}


def test_split_is_deterministic_in_the_seed():
    rows = _corpus()
    a = P.build_splits(rows, seed=7, n_train=300, n_dev=120, held_out=("trec",))
    b = P.build_splits(rows, seed=7, n_train=300, n_dev=120, held_out=("trec",))
    c = P.build_splits(rows, seed=8, n_train=300, n_dev=120, held_out=("trec",))
    assert a == b and a["train"] != c["train"]


# ---- encode -------------------------------------------------------------------------
class FakeTok:
    """Whitespace tokenizer; marker tokens get fixed ids like the real Qwen tokenizer."""
    specials = {"<|fim_prefix|>": 900, "<|fim_middle|>": 901, "<|box_start|>": 902, "<|box_end|>": 903, "<|fim_suffix|>": 904}

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 800 for w in text.split()]

    def convert_tokens_to_ids(self, tok):
        return self.specials[tok]


def test_encode_marks_decide_and_option_end_positions():
    row = P.flatten([_record("r1", "yelp", state="w " * 10)])[0]  # choice, 3 options
    enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, shuffle=False)
    ids = enc["ids"]
    assert ids[enc["decide_pos"]] == 904
    assert len(enc["opt_pos"]) == 3 and all(ids[p] == 903 for p in enc["opt_pos"])
    assert ids[0] == 900 and enc["decide_pos"] == len(ids) - 1
    assert enc["label"] == 1 and enc["qtype"] == "choice" and enc["n_options"] == 3


def test_encode_shuffle_permutes_options_and_label_together():
    row = P.flatten([_record("r1", "yelp")])[0]
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(20):
        enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, shuffle=True, rng=rng)
        seen.add(enc["label"])
        assert enc["perm"][enc["label"]] == row["label"]     # the label follows its option
    assert len(seen) > 1


def test_encode_never_shuffles_noul_or_score():
    rows = P.flatten([_record("r1", "yelp")])
    rng = np.random.default_rng(0)
    for row in rows[1:]:
        for _ in range(10):
            enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, shuffle=True, rng=rng)
            assert enc["perm"] == list(range(row["n_options"])) and enc["label"] == row["label"]


def test_encode_truncates_long_state_to_budget():
    row = P.flatten([_record("r1", "yelp", state="w " * 5000)])[0]
    enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, shuffle=False)
    assert len(enc["ids"]) <= P.MAX_ROW_TOKENS
    assert enc["state_tokens"] == P.MAX_STATE_TOKENS


def test_flatten_renders_dict_instructions_and_list_state_as_text():
    rec = _record("r1", "banking77", state=[{"role": "customer", "content": "hi"}])
    rec["questions"]["q_choice"]["instructions"] = {"question": "Which intent?", "focus": "Whole message."}
    row = P.flatten([rec])[0]
    assert isinstance(row["instructions"], str) and "Which intent?" in row["instructions"] and "Whole message." in row["instructions"]
    assert isinstance(row["state"], str) and "customer" in row["state"]
