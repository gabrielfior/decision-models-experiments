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


# ---- per-torso layout ------------------------------------------------------------------
def test_torso_config_gives_markers_and_tap_layers_for_each_ladder_torso():
    q = P.torso_config("Qwen/Qwen3.5-0.8B-Base")
    assert q["markers"] == P.QWEN_MARKERS and q["tap_layers"] == (12, 16, 20, 24)
    l = P.torso_config("LiquidAI/LFM2.5-230M-Base")
    assert l["tap_layers"] == (7, 9, 12, 14) and l["markers"]["decide"] == "<|fim_suf|>"
    m = P.torso_config("answerdotai/ModernBERT-large")
    assert m["tap_layers"] == (14, 19, 23, 28) and m["markers"]["opt_end"] == "[unused3]"
    assert P.torso_config("Qwen3.5-0.8B-Base") == q          # short name works too


def test_batches_use_the_torso_markers():
    class LfmTok(FakeTok):
        specials = {"<|fim_pre|>": 3, "<|fim_mid|>": 4, "<|tool_list_start|>": 8, "<|tool_list_end|>": 9, "<|fim_suf|>": 5}
    rows = [r for r in P.flatten([_record("r1", "yelp")]) if r["qtype"] == "choice"]
    markers = P.torso_config("LiquidAI/LFM2.5-230M-Base")["markers"]
    (b,) = list(P.batches(LfmTok(), rows, 1, shuffle_options=False, device="cpu", markers=markers))
    assert b["ids"][0, b["decide_pos"][0]].item() == 5
    assert all(b["ids"][0, p].item() == 9 for p in b["opt_pos"][0])


def test_encode_truncates_long_instructions_so_options_keep_their_text():
    rec = _record("r1", "yelp", state="s")
    rec["questions"]["q_choice"]["instructions"] = "w " * 5000
    row = P.flatten([rec])[0]
    enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, shuffle=False)
    assert len(enc["ids"]) <= P.MAX_ROW_TOKENS
    assert enc["instr_tokens"] == P.MAX_INSTR_TOKENS
    # each option still has its own text (not capped to 1 token)
    assert enc["opt_pos"][1] - enc["opt_pos"][0] > 2


def test_torso_config_carries_lora_target_modules_per_architecture():
    q = P.torso_config("Qwen/Qwen3.5-0.8B-Base")["lora_targets"]
    assert "q_proj" in q and "in_proj_qkv" in q and "gate_proj" in q          # attention, GatedDeltaNet, MLP
    l = P.torso_config("LiquidAI/LFM2.5-230M-Base")["lora_targets"]
    assert set(l) == {"q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"}
    m = P.torso_config("answerdotai/ModernBERT-large")["lora_targets"]
    assert set(m) == {"Wqkv", "Wo", "Wi"}


def test_qwen_2b_shares_the_qwen_layout_for_the_scaling_check():
    c = P.torso_config("Qwen/Qwen3.5-2B-Base")
    assert c["markers"] == P.QWEN_MARKERS and c["tap_layers"] == (12, 16, 20, 24) and c["n_layers"] == 24
    assert c["lora_targets"] == P.torso_config("Qwen/Qwen3.5-0.8B-Base")["lora_targets"]


def test_encode_row_cyclic_shift_orders_rotate_choice_options_and_the_label():
    row = P.flatten([_record("r1", "yelp")])[0]            # choice, options a b c, label index 1
    enc = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, order="shift:1")
    assert enc["perm"] == [1, 2, 0] and enc["label"] == 0     # shown first is canonical option 1 (the label)
    enc2 = P.encode_row(FakeTok(), row, P.QWEN_MARKERS, order="shift:2")
    assert enc2["perm"] == [2, 0, 1] and enc2["label"] == 2
    noul = P.flatten([_record("r1", "yelp")])[1]
    assert P.encode_row(FakeTok(), noul, P.QWEN_MARKERS, order="shift:1")["perm"] == [0, 1]   # never reordered
