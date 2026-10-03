"""Tests for the frozen training/eval harness in prepare.py (sections 5–6), CPU only."""
import csv
import json
import pytest

import numpy as np
import torch

import prepare as P
from tests.test_data import FakeTok, _record


# ---- batching -----------------------------------------------------------------------
def test_batches_pad_rows_and_keep_positions_aligned():
    rows = P.flatten([_record("r1", "yelp", state="a b c"), _record("r2", "yelp", state="a " * 40)])
    rows = [r for r in rows if r["qtype"] == "choice"]           # 3 options each, different lengths
    (b,) = list(P.batches(FakeTok(), rows, batch_size=2, shuffle_options=False, device="cpu"))
    B, T = b["ids"].shape
    assert B == 2 and b["attn"].shape == (2, T) and b["opt_mask"].shape == (2, 3)
    for i in range(2):
        assert b["ids"][i, b["decide_pos"][i]].item() == 904
        assert all(b["ids"][i, p].item() == 903 for p in b["opt_pos"][i])
        assert b["attn"][i].sum().item() == b["decide_pos"][i].item() + 1   # right-padded
    assert b["label"].tolist() == [1, 1]


# ---- temperature --------------------------------------------------------------------
def _preds(n, qtype, scale, rng):
    """Synthetic predictions: logits whose correct option is ahead by `scale`."""
    out = []
    for _ in range(n):
        k = 3
        z = rng.normal(size=k)
        y = rng.integers(k)
        z[y] += scale
        out.append({"logits": z, "label": int(y), "qtype": qtype, "n_options": k, "id": "x"})
    return out


def test_fit_temperature_sharpens_underconfident_and_flattens_overconfident():
    rng = np.random.default_rng(0)
    under = _preds(2000, "choice", 6.0, rng)                 # big margin, honest -> T < 1 is NOT wanted...
    for p in under:
        p["logits"] = p["logits"] * 0.2                      # ...so shrink logits: now underconfident
    over = _preds(2000, "noul", 0.5, rng)
    for p in over:
        p["logits"] = p["logits"] * 5.0                      # overconfident
    temps = P.fit_temperature_by_type(under + over)
    assert temps["choice"] < 1.0 and temps["noul"] > 1.0


def test_temperature_does_not_change_the_argmax():
    rng = np.random.default_rng(1)
    preds = _preds(50, "score", 1.0, rng)
    temps = {"score": 2.7}
    probs = P.probs_from(preds, temps)
    for p, pr in zip(preds, probs):
        assert int(np.argmax(pr)) == int(np.argmax(p["logits"]))
        assert abs(pr.sum() - 1) < 1e-9


# ---- evaluate -----------------------------------------------------------------------
def test_evaluate_reports_dev_heldout_and_order_sensitivity():
    rng = np.random.default_rng(2)
    dev_a = _preds(300, "choice", 2.0, rng)
    dev_b = [dict(p) for p in dev_a]
    for p in dev_b[:30]:                                     # 10% of rows flip under the other order
        p["logits"] = p["logits"][::-1].copy()
    held = _preds(100, "noul", 2.0, rng)
    rep = P.evaluate(dev_a, dev_b, held, temps={"choice": 1.0, "noul": 1.0, "score": 1.0})
    for k in ("dev_selection", "dev_se", "dev_acc", "dev_brier", "dev_ece", "heldout_acc", "heldout_brier", "order_sens"):
        assert k in rep
    assert 0.0 < rep["order_sens"] <= 0.1 + 1e-9
    assert rep["dev"]["per_type"]["choice"]["n"] == 300 and rep["heldout"]["per_type"]["noul"]["n"] == 100


# ---- results.tsv --------------------------------------------------------------------
def test_append_results_writes_one_row_with_the_program_columns(tmp_path):
    path = tmp_path / "results.tsv"
    rep = {"tier": "A", "note": "unit test", "seeds": 3, "dev_selection": 0.5, "dev_se": 0.01, "dev_acc": 0.7,
           "dev_brier": 0.4, "dev_ece": 0.05, "heldout_acc": 0.6, "heldout_brier": 0.5, "order_sens": 0.01,
           "p50_ms": 1.5, "kept": True}
    P.append_results(rep, path=path, commit="abc1234")
    P.append_results(rep, path=path, commit="abc1234")
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 2 and rows[0]["commit"] == "abc1234" and rows[0]["kept"] == "True"
    assert list(rows[0].keys()) == P.RESULT_COLUMNS and rows[0]["per_seed"] == ""


# ---- Tier A on a synthetic cache ----------------------------------------------------
def _synthetic_cache(path, n, d=16, seed=0):
    """Random read-out vectors with the answer planted: h_opt[label] ~= h_ans at every layer."""
    rng = np.random.default_rng(seed)
    L = len(P.TAP_LAYERS)
    qtypes = np.array(P.QTYPES)[rng.integers(3, size=n)]
    n_opts = np.where(qtypes == "noul", 2, rng.integers(2, 6, size=n))
    labels = (rng.random(n) * n_opts).astype(int)
    h_ans = rng.normal(size=(n, L, d)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(n_opts)])
    h_opts = rng.normal(size=(offsets[-1], L, d)).astype(np.float16)
    for i in range(n):
        h_opts[offsets[i] + labels[i]] = h_ans[i] + 0.3 * rng.normal(size=(L, d))
    np.savez(path, h_ans=h_ans, h_opts=h_opts, offsets=offsets, labels=labels, qtypes=qtypes,
             n_options=n_opts, ids=np.array([f"r{i}" for i in range(n)]), d_model=d)


def test_tier_a_trains_head_on_cache_and_beats_chance(tmp_path):
    for name, n in (("train", 600), ("dev", 300), ("heldout", 100)):
        _synthetic_cache(tmp_path / f"{name}.npz", n, seed=hash(name) % 1000)
    import head as head_mod
    rep = P.tier_a(head_mod, cache_dir=tmp_path, seeds=(0, 1), epochs=30, note="synthetic", results_path=tmp_path / "r.tsv",
                   baseline=None)
    assert rep["seeds"] == 2 and rep["dev_acc"] > 0.8
    assert rep["kept"] is True                      # no baseline: the first result is always kept
    assert (tmp_path / "r.tsv").exists()


def test_read_baseline_can_pin_a_kept_row_by_note_substring(tmp_path):
    path = tmp_path / "r.tsv"
    for note, sel, seeds in (("plain pointer", 0.28, [0.3, 0.29, 0.25]), ("LN pointer layer 16", 0.44, [0.42, 0.45, 0.44]), ("later thing", 0.46, [0.46, 0.46, 0.46])):
        P.append_results({"tier": "A", "note": note, "seeds": 3, "dev_selection": sel, "kept": True, "per_seed": seeds}, path=path, commit="x")
    assert P.read_baseline(path) == [0.46, 0.46, 0.46]
    assert P.read_baseline(path, match="LN pointer") == [0.42, 0.45, 0.44]
    assert P.read_baseline(path, match="nope") is None


def test_ungated_rows_do_not_become_the_baseline_once_a_kept_row_exists(tmp_path):
    for name, n in (("train", 300), ("dev", 150), ("heldout", 60)):
        _synthetic_cache(tmp_path / f"{name}.npz", n, seed=n)
    import head as head_mod
    p = tmp_path / "r.tsv"
    first = P.tier_a(head_mod, cache_dir=tmp_path, seeds=(0,), epochs=5, note="first", results_path=p, baseline=None)
    assert first["kept"] is True                                   # the very first row anchors the file
    explore = P.tier_a(head_mod, cache_dir=tmp_path, seeds=(0,), epochs=1, note="exploratory", results_path=p, baseline=None)
    assert explore["kept"] == "ungated"
    assert P.read_baseline(p) == pytest.approx(first["per_seed"], abs=1e-4)   # still the first row, not the exploratory one


def test_tier_a_saves_the_best_head_with_its_config(tmp_path):
    for name, n in (("train", 200), ("dev", 100), ("heldout", 40)):
        _synthetic_cache(tmp_path / f"{name}.npz", n, seed=n)
    import head as head_mod
    rep = P.tier_a(head_mod, cache_dir=tmp_path, seeds=(0, 1), epochs=3, note="save me", results_path=tmp_path / "r.tsv",
                   baseline=None, save_dir=tmp_path / "out")
    assert (tmp_path / "out" / "head.pt").exists()
    cfg = json.loads((tmp_path / "out" / "head.json").read_text())
    assert cfg["head"] == dict(head_mod.HEAD_CONFIG) and cfg["seed"] in (0, 1) and "temps" in cfg


def test_code_hash_changes_when_an_editable_file_changes(tmp_path, monkeypatch):
    a = P.code_hash()
    monkeypatch.setattr(P, "EDITABLE_FILES", [tmp_path / "x.py"])
    (tmp_path / "x.py").write_text("one")
    b = P.code_hash()
    (tmp_path / "x.py").write_text("two")
    assert a != b and b != P.code_hash() and len(b) == 8
