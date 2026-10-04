"""Abstain / unrelated-label augmentation (prepare.augment_row; train --augment)."""
import copy

import numpy as np

import prepare as P
import train
from tests.test_data import _record

ABSTAIN = ("none of the above", None)


def _choice(rid="r1", crit=None, label="b", source="yelp"):
    rec = _record(rid, source)
    if crit is not None:
        rec["questions"]["q_choice"]["criteria"] = crit
        rec["questions"]["q_choice"]["label"] = label
    return P.flatten([rec])[0]


def _pool():
    return [("x", "X desc"), ("y", None), ("z", "Z desc"), ("w", "W"), ("v", "V")]


def test_noul_and_score_rows_are_returned_unchanged():
    rows = P.flatten([_record("r1", "yelp")])
    rng = np.random.default_rng(0)
    for row in rows[1:]:
        before = copy.deepcopy(row)
        out = P.augment_row(row, rng, p_abstain=1.0, p_unrelated=1.0, pool=_pool())
        assert out == before and row == before


def test_choice_rows_with_fewer_than_three_options_are_returned_unchanged():
    row = _choice(crit={"a": "A", "b": "B"}, label="a")
    before = copy.deepcopy(row)
    out = P.augment_row(row, np.random.default_rng(0), p_abstain=1.0, p_unrelated=1.0, pool=_pool())
    assert out == before and row == before


def test_abstain_appends_one_trailing_option_and_keeps_the_label():
    row = _choice()
    before = copy.deepcopy(row)
    out = P.augment_row(row, np.random.default_rng(0), p_abstain=1.0, p_unrelated=0.0, pool=_pool())
    assert out["options"] == before["options"] + [ABSTAIN]
    assert out["label"] == before["label"] == 1
    assert out["n_options"] == before["n_options"] + 1 == 4
    assert out["qtype"] == "choice" and out["id"] == before["id"] and out["state"] == before["state"]
    assert row == before                      # never mutated in place
    assert out is not row and out["options"] is not row["options"]


def test_unrelated_replaces_texts_from_the_pool_and_moves_the_label_to_the_abstain_slot():
    row = _choice()
    before = copy.deepcopy(row)
    pool = _pool() + list(before["options"])           # pool may contain this row's own labels: they must not be drawn
    out = P.augment_row(row, np.random.default_rng(1), p_abstain=1.0, p_unrelated=1.0, pool=pool)
    n = before["n_options"]
    assert out["n_options"] == n + 1 and len(out["options"]) == n + 1
    assert out["options"][-1] == ABSTAIN
    assert out["label"] == n                            # the abstain option is the answer
    drawn = out["options"][:-1]
    assert all(o in _pool() for o in drawn)
    assert not any(o in before["options"] for o in drawn)
    assert not any(o[0] in {k for k, _ in before["options"]} for o in drawn)
    assert len({k for k, _ in drawn}) == n              # distinct keys
    assert row == before


def test_zero_probabilities_give_an_identical_copy():
    row = _choice()
    before = copy.deepcopy(row)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = P.augment_row(row, rng, p_abstain=0.0, p_unrelated=0.0, pool=_pool())
        assert out == before
    out = P.augment_row(row, rng, p_abstain=0.0, p_unrelated=1.0, pool=_pool())
    assert out == before and row == before


def test_abstain_only_when_the_pool_is_missing_or_too_small():
    row = _choice()
    for pool in (None, [], [("x", "X")], [("x", "X"), ("a", "A")]):   # "a" is one of the row's own keys
        out = P.augment_row(row, np.random.default_rng(0), p_abstain=1.0, p_unrelated=1.0, pool=pool)
        assert out["options"] == row["options"] + [ABSTAIN] and out["label"] == row["label"]


def test_augmentation_is_deterministic_under_a_seeded_rng_and_mixes_both_kinds():
    rows = [_choice(rid=f"r{i}") for i in range(300)]
    pool = _pool()
    outs = []
    for seed in (7, 7):
        rng = np.random.default_rng(seed)
        outs.append([P.augment_row(r, rng, pool=pool) for r in rows])
    assert outs[0] == outs[1]
    kinds = {"plain": 0, "abstain": 0, "unrelated": 0}
    for r, o in zip(rows, outs[0]):
        if o == r:
            kinds["plain"] += 1
        elif o["options"][:-1] == r["options"]:
            kinds["abstain"] += 1
        else:
            kinds["unrelated"] += 1
    assert kinds["plain"] > 200 and kinds["abstain"] > 0 and kinds["unrelated"] > 0   # p_abstain .10, p_unrelated .25


# ---- train.py side: the pool and the per-epoch wrap -----------------------------------
def test_augment_pool_collects_unique_choice_options_only():
    rows = P.flatten([_record("r1", "yelp"), _record("r2", "imdb")])
    rows.append(_choice(rid="r3", crit={"d": "D", "e": None, "a": "A"}, label="d"))
    pool = train.augment_pool(rows)
    assert sorted(pool, key=repr) == pool and len(pool) == len(set(pool))
    assert set(pool) == {("a", "A"), ("b", None), ("c", "C"), ("d", "D"), ("e", None)}
    assert ("yes", "pos") not in pool and ("0", "1 star") not in pool     # noul / score never enter the pool


def test_epoch_rows_is_the_identity_when_augment_is_off_and_redraws_per_epoch_when_on():
    rows = [_choice(rid=f"r{i}") for i in range(200)]
    rng = np.random.default_rng(0)
    assert train.epoch_rows(rows, rng, augment=False, pool=None) is rows
    pool = train.augment_pool(rows)
    e1 = train.epoch_rows(rows, rng, augment=True, pool=pool)
    e2 = train.epoch_rows(rows, rng, augment=True, pool=pool)
    assert len(e1) == len(e2) == len(rows) and [r["id"] for r in e1] == [r["id"] for r in rows]
    assert e1 != rows and e1 != e2                     # augmented, and a different draw each epoch
    assert all(r["n_options"] in (3, 4) for r in e1)


def test_augment_flag_is_off_by_default():
    ap = train.build_parser()
    args = ap.parse_args(["--note", "x"])
    assert args.augment is False
    assert ap.parse_args(["--note", "x", "--augment"]).augment is True
