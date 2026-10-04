"""Tests for the JevBench-facing pieces: question -> row, label-free encoding, answer format."""
import numpy as np
import pytest

import prepare as P
import serve
from tests.test_data import FakeTok


def test_question_to_row_matches_flatten_conventions():
    q = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": None}}
    row = P.question_to_row("state text", q)
    assert row["options"] == [("a", "A"), ("b", None)] and row["label"] is None and row["qtype"] == "choice"
    noul = P.question_to_row("s", {"type": "noul", "instructions": "?", "criteria": None})
    assert noul["options"] == [("no", None), ("yes", None)]
    score = P.question_to_row({"k": "v"}, {"type": "score", "instructions": "?", "criteria": ["low", "high"]})
    assert score["options"] == [("0", "low"), ("1", "high")] and score["state"] == "k: v"


def test_encode_row_accepts_missing_label():
    row = P.question_to_row("s", {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}})
    enc = P.encode_row(FakeTok(), row, shuffle=False)
    assert enc["label"] is None and enc["n_options"] == 2


def test_build_answer_formats_each_question_type_for_the_typesafe_adapter():
    noul = serve.build_answer("noul", [("no", None), ("yes", None)], np.array([0.3, 0.7]))
    assert noul == {"type": "noul", "noul": pytest.approx(0.7)}
    choice = serve.build_answer("choice", [("a", None), ("b", None), ("c", None)], np.array([0.2, 0.5, 0.3]))
    assert choice["type"] == "choice" and choice["choice"] == "b"
    assert choice["probabilities"] == {"a": pytest.approx(0.2), "b": pytest.approx(0.5), "c": pytest.approx(0.3)}
    score = serve.build_answer("score", [("0", "x"), ("1", "y")], np.array([0.9, 0.1]))
    assert score == {"type": "score", "probabilities": {"0": pytest.approx(0.9), "1": pytest.approx(0.1)}}


def test_build_answer_probabilities_sum_to_one_within_harness_tolerance():
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    ans = serve.build_answer("choice", [("a", None), ("b", None), ("c", None)], p)
    assert abs(sum(ans["probabilities"].values()) - 1) < 1e-3


def test_saved_head_config_does_not_inherit_newer_defaults():
    # a run saved before the "norm" key existed must rebuild as the head it was trained as
    cfg = serve.head_config_from_run({"name": "pointer", "width": 256, "layer": -1})
    assert cfg["norm"] is False and cfg["layer"] == -1
    assert serve.head_config_from_run(None) is None


def test_canonical_probs_undo_the_option_permutation_and_tta_averages():
    z = np.array([2.0, 0.0, 1.0])                 # logits in presented order
    perm = [2, 0, 1]                              # presented position j shows canonical option perm[j]
    p = serve.canonical_probs(z, perm)
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-9
    assert p[2] > p[1] > p[0]                     # canonical option 2 was shown first with the top logit
    avg = serve.average_probs([p, np.array([0.2, 0.3, 0.5])])
    assert abs(avg.sum() - 1) < 1e-9 and np.argmax(avg) == 2


def test_tta_orders_lists_distinct_orders_up_to_n():
    assert serve.tta_orders(1) == ["identity"]
    assert serve.tta_orders(2) == ["identity", "reversed"]
    assert serve.tta_orders(4) == ["identity", "reversed", "shift:1", "shift:2"]
    assert len(set(serve.tta_orders(8))) == 8


def test_complete_head_cfg_adds_letter_emb_and_turns_on_the_prefix_only_for_hybrid():
    import torch
    from tests.test_harness import _FakeTorso
    torso, tok = _FakeTorso(), FakeTok()
    cfg, prefix = serve.complete_head_cfg({"name": "hybrid", "width": 32, "layer": -1, "norm": True}, torso, tok)
    assert prefix is True and cfg["name"] == "hybrid" and cfg["letter_emb"].shape == (26, 8)
    assert torch.allclose(cfg["letter_emb"], torso.emb.weight[P.letter_token_ids(tok)].detach())
    cfg2, prefix2 = serve.complete_head_cfg({"name": "pointer", "width": 32, "layer": -1, "norm": True}, torso, tok)
    assert prefix2 is False and "letter_emb" not in cfg2
    assert serve.complete_head_cfg(None, torso, tok) == (None, False)


def test_option_preview_from_run_config_turns_on_the_module_constant(monkeypatch):
    monkeypatch.setattr(P, "OPTION_PREVIEW", False)
    assert serve.option_preview_from_run({}) is False and P.OPTION_PREVIEW is False
    assert serve.option_preview_from_run({"option_preview": False}) is False and P.OPTION_PREVIEW is False
    assert serve.option_preview_from_run({"option_preview": True}) is True and P.OPTION_PREVIEW is True
