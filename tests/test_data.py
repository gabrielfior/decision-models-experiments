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


# ---- letter prefix (hybrid read-out) -------------------------------------------------
def _option_segments(enc):
    """Token ids between each [opt] and its [opt_end], in presented order."""
    ids = enc["ids"]
    starts = [i for i, t in enumerate(ids) if t == 902]
    return [ids[s + 1:e] for s, e in zip(starts, enc["opt_pos"])]


def test_letter_token_ids_gives_one_id_per_letter_and_falls_back_to_the_first_token():
    ids = P.letter_token_ids(FakeTok())
    assert len(ids) == 26 and len(P.LETTERS) == 26 and P.LETTERS[0] == "A" and P.LETTERS[-1] == "Z"
    assert ids == [FakeTok().encode(" " + c)[0] for c in P.LETTERS]

    class SplitTok(FakeTok):                      # a tokenizer that needs two tokens per letter
        def encode(self, text, add_special_tokens=False):
            return [7, 8]
    assert P.letter_token_ids(SplitTok()) == [7] * 26


def test_encode_row_letter_prefix_labels_options_by_presented_position():
    row = P.flatten([_record("r1", "yelp")])[0]          # choice, options a b c
    tok = FakeTok()
    plain = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=False)
    for seg, (key, desc) in zip(_option_segments(plain), row["options"]):
        assert seg == tok.encode(P.option_text("choice", key, desc))      # default: no prefix
    rng = np.random.default_rng(5)
    seen = set()
    for _ in range(20):
        enc = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=True, rng=rng, letter_prefix=True)
        seen.add(tuple(enc["perm"]))
        for j, seg in enumerate(_option_segments(enc)):
            key, desc = row["options"][enc["perm"][j]]
            assert seg == tok.encode(f"{P.LETTERS[j]}) " + P.option_text("choice", key, desc))   # letter follows the slot
        assert enc["perm"][enc["label"]] == row["label"]
    assert len(seen) > 1
    assert P.LETTER_PREFIX is False                        # additive option, default off


def test_letter_prefix_default_follows_the_module_constant(monkeypatch):
    row = P.flatten([_record("r1", "yelp")])[0]
    tok = FakeTok()
    monkeypatch.setattr(P, "LETTER_PREFIX", True)
    enc = P.encode_row(tok, row, P.QWEN_MARKERS)
    assert _option_segments(enc)[0] == tok.encode("A) " + P.option_text("choice", *row["options"][0]))
    off = P.encode_row(tok, row, P.QWEN_MARKERS, letter_prefix=False)
    assert _option_segments(off)[0] == tok.encode(P.option_text("choice", *row["options"][0]))
    (b,) = list(P.batches(tok, [row], 1, shuffle_options=False, device="cpu", letter_prefix=False))
    assert b["ids"].shape[1] == len(off["ids"])


# ---- option preview (options visible while the state is read) -------------------------
def test_option_preview_is_off_by_default_and_leaves_ids_identical():
    assert P.OPTION_PREVIEW is False
    tok = FakeTok()
    for row in P.flatten([_record("r1", "yelp", state="w " * 10)]):
        plain = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=False)
        off = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=False, option_preview=False)
        assert plain == off
        assert plain["ids"][1:1 + plain["state_tokens"]] == tok.encode(row["state"])     # state right after [state]


def test_option_preview_ids_compose_prefix_truncated_options_and_separators_in_presented_order():
    tok = FakeTok()
    row = P.flatten([_record("r1", "yelp")])[0]                  # choice: a: A, b (no desc), c: C
    pv = P.option_preview_ids(tok, row["qtype"], row["options"], [2, 0, 1])
    sep = tok.encode(P.OPTION_PREVIEW_SEP)
    expected = tok.encode(P.OPTION_PREVIEW_PREFIX)
    for j, k in enumerate([2, 0, 1]):
        if j:
            expected += sep
        expected += tok.encode(P.option_text("choice", *row["options"][k]))[:P.OPTION_PREVIEW_TOKENS]
    assert pv == expected
    # the raw key stands in when the description is None
    assert tok.encode("b") == tok.encode(P.option_text("choice", "b", None))
    # long option texts are cut to OPTION_PREVIEW_TOKENS tokens each
    long_options = [("a", "x " * 30), ("b", "y " * 30)]
    pv2 = P.option_preview_ids(tok, "choice", long_options, [0, 1])
    assert len(pv2) == len(tok.encode(P.OPTION_PREVIEW_PREFIX)) + 2 * P.OPTION_PREVIEW_TOKENS + len(sep)


def test_encode_row_option_preview_sits_between_state_marker_and_state_text():
    tok = FakeTok()
    row = P.flatten([_record("r1", "yelp", state="w " * 10)])[0]
    plain = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=False)
    enc = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=False, option_preview=True)
    pv = P.option_preview_ids(tok, row["qtype"], row["options"], enc["perm"])
    assert pv and enc["ids"][0] == 900
    assert enc["ids"][1:1 + len(pv)] == pv                                            # preview after [state]
    assert enc["ids"][1 + len(pv):1 + len(pv) + enc["state_tokens"]] == tok.encode(row["state"])   # then the state
    assert enc["ids"][1 + len(pv) + enc["state_tokens"]] == 901                        # then [q]
    # everything after the preview is the plain layout, shifted
    assert enc["ids"] == plain["ids"][:1] + pv + plain["ids"][1:]
    assert enc["decide_pos"] == plain["decide_pos"] + len(pv) and enc["ids"][enc["decide_pos"]] == 904
    assert enc["opt_pos"] == [p + len(pv) for p in plain["opt_pos"]] and all(enc["ids"][p] == 903 for p in enc["opt_pos"])
    assert enc["label"] == plain["label"] and enc["perm"] == plain["perm"]


def test_encode_row_option_preview_follows_the_shuffle_permutation():
    tok = FakeTok()
    row = P.flatten([_record("r1", "yelp")])[0]
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(20):
        enc = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=True, rng=rng, option_preview=True)
        seen.add(tuple(enc["perm"]))
        pv = P.option_preview_ids(tok, "choice", row["options"], enc["perm"])
        assert enc["ids"][1:1 + len(pv)] == pv
        # the full option list agrees with the preview: same permutation
        for j, seg in enumerate(_option_segments(enc)):
            assert seg == tok.encode(P.option_text("choice", *row["options"][enc["perm"][j]]))
        assert enc["perm"][enc["label"]] == row["label"]
    assert len(seen) > 1
    # the preview differs between two different permutations
    a = P.option_preview_ids(tok, "choice", row["options"], [0, 1, 2])
    b = P.option_preview_ids(tok, "choice", row["options"], [2, 1, 0])
    assert a != b


def test_encode_row_option_preview_on_noul_and_score_keeps_the_fixed_order():
    tok = FakeTok()
    rows = P.flatten([_record("r1", "yelp")])
    rng = np.random.default_rng(0)
    for row in rows[1:]:                                           # noul, score
        enc = P.encode_row(tok, row, P.QWEN_MARKERS, shuffle=True, rng=rng, option_preview=True)
        pv = P.option_preview_ids(tok, row["qtype"], row["options"], list(range(row["n_options"])))
        assert enc["perm"] == list(range(row["n_options"])) and enc["label"] == row["label"]
        assert enc["ids"][1:1 + len(pv)] == pv
        assert enc["ids"][enc["decide_pos"]] == 904 and all(enc["ids"][p] == 903 for p in enc["opt_pos"])
    # score options show their level description, not the index
    score = rows[2]
    pv = P.option_preview_ids(tok, "score", score["options"], [0, 1, 2])
    assert pv[len(tok.encode(P.OPTION_PREVIEW_PREFIX)):][:2] == tok.encode("1 star")


def test_option_preview_default_follows_the_module_constant(monkeypatch):
    tok = FakeTok()
    row = P.flatten([_record("r1", "yelp")])[0]
    monkeypatch.setattr(P, "OPTION_PREVIEW", True)
    on = P.encode_row(tok, row, P.QWEN_MARKERS)
    pv = P.option_preview_ids(tok, "choice", row["options"], on["perm"])
    assert on["ids"][1:1 + len(pv)] == pv
    off = P.encode_row(tok, row, P.QWEN_MARKERS, option_preview=False)
    assert off["ids"][1:1 + off["state_tokens"]] == tok.encode(row["state"])
    (b,) = list(P.batches(tok, [row], 1, shuffle_options=False, device="cpu", option_preview=False))
    assert b["ids"].shape[1] == len(off["ids"])
    (b_on,) = list(P.batches(tok, [row], 1, shuffle_options=False, device="cpu"))
    assert b_on["ids"].shape[1] == len(on["ids"]) == len(off["ids"]) + len(pv)
    assert b_on["ids"][0, b_on["decide_pos"][0]].item() == 904


def test_option_preview_counts_in_the_row_budget_and_truncates_the_state_first():
    tok = FakeTok()
    # a long state alone is cut to MAX_STATE_TOKENS; preview included, the row still fits
    row = P.flatten([_record("r1", "yelp", state="w " * 5000)])[0]
    enc = P.encode_row(tok, row, P.QWEN_MARKERS, option_preview=True)
    assert len(enc["ids"]) <= P.MAX_ROW_TOKENS and enc["state_tokens"] == P.MAX_STATE_TOKENS
    # huge options overflow the budget: with the preview on, the state yields before the options are capped
    rec = _record("r1", "yelp", state="s " * 300)
    rec["questions"]["q_choice"]["criteria"] = {k: f"{k} " * 700 for k in "abc"}
    row = P.flatten([rec])[0]
    plain = P.encode_row(tok, row, P.QWEN_MARKERS)
    enc = P.encode_row(tok, row, P.QWEN_MARKERS, option_preview=True)
    assert len(plain["ids"]) <= P.MAX_ROW_TOKENS and len(enc["ids"]) <= P.MAX_ROW_TOKENS
    assert plain["state_tokens"] == 300                                   # default path untouched
    assert P.OPTION_PREVIEW_MIN_STATE <= enc["state_tokens"] < 300        # the state gave way first
    pv = P.option_preview_ids(tok, "choice", row["options"], enc["perm"])
    assert enc["ids"][1:1 + len(pv)] == pv and enc["ids"][enc["decide_pos"]] == 904
    assert all(enc["ids"][p] == 903 for p in enc["opt_pos"])
    # the options kept at least as much text as they had without the preview
    assert all(len(b) >= len(a) for a, b in zip(_option_segments(plain), _option_segments(enc)))
