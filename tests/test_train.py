"""CPU tests for the pure parts of train.py."""
import numpy as np
import torch

import prepare as P
import train
from tests.test_data import FakeTok, _record


def test_gather_readout_picks_decide_and_option_positions_per_layer():
    B, T, d = 2, 9, 3
    n_layers = max(P.TAP_LAYERS) + 1
    # hidden state at (layer l, position t) encodes l and t so we can check the pick
    pos = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, d)
    hs = tuple((pos + l * 100).clone() for l in range(n_layers))
    decide_pos = torch.tensor([8, 5])
    opt_pos = torch.tensor([[2, 4, 6], [1, 3, -1]])
    mask = opt_pos >= 0
    h_ans, h_opts = train.gather_readout(hs, decide_pos, opt_pos, mask)
    assert h_ans.shape == (B, len(P.TAP_LAYERS), d) and h_opts.shape == (B, len(P.TAP_LAYERS), 3, d)
    for li, l in enumerate(P.TAP_LAYERS):
        assert h_ans[0, li, 0].item() == l * 100 + 8 and h_ans[1, li, 0].item() == l * 100 + 5
        assert h_opts[0, li, 1, 0].item() == l * 100 + 4 and h_opts[1, li, 0, 0].item() == l * 100 + 1
        assert h_opts[1, li, 2, 0].item() == 0.0          # masked option zeroed


def test_head_configs_record_the_layer_each_depth_head_reads():
    cfgs = train.head_configs(n_layers=4, depth_heads=True)
    assert [c["layer"] for c in cfgs] == [0, 1, 2, 3]
    single = train.head_configs(n_layers=4, depth_heads=False)
    assert len(single) == 1 and single[0] == dict(train.head_mod.HEAD_CONFIG)


def test_head_overrides_change_only_the_given_keys(monkeypatch):
    monkeypatch.setattr(train.head_mod, "HEAD_CONFIG", dict(train.head_mod.HEAD_CONFIG))
    train.apply_head_overrides('{"name": "cross_option", "residual_pointer": true}')
    assert train.head_mod.HEAD_CONFIG["name"] == "cross_option" and train.head_mod.HEAD_CONFIG["residual_pointer"] is True
    assert train.head_mod.HEAD_CONFIG["layer"] == 1 and train.head_mod.HEAD_CONFIG["norm"] is True


def test_split_plan_defaults_and_final_retrain_mode():
    assert train.split_plan("train", "dev") == (["train"], "dev", ["heldout"])
    tr, calib, report = train.split_plan("train,dev", "heldout")
    assert tr == ["train", "dev"] and calib == "heldout" and report == []


# ---- option-shuffle consistency loss (--consistency W) -------------------------------
def _presented(canon, perm):
    """Logits as the head emits them when presented position j shows canonical option perm[j]."""
    return torch.tensor([[canon[i][j] for j in p] for i, p in enumerate(perm)], dtype=torch.float32)


def test_consistency_loss_is_zero_for_same_distribution_under_different_perms():
    canon = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]
    perm1, perm2 = [[0, 1, 2], [2, 0, 1]], [[2, 0, 1], [1, 2, 0]]
    z1, z2 = _presented(canon, perm1), _presented(canon, perm2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss = train.consistency_loss(z1, perm1, z2, perm2, mask, mask)
    assert loss.shape == () and abs(loss.item()) < 1e-6


def test_consistency_loss_is_positive_and_symmetric_when_distributions_differ():
    perm1, perm2 = [[0, 1, 2]], [[1, 2, 0]]
    z1 = _presented([[1.0, 2.0, 3.0]], perm1)
    z2 = _presented([[3.0, 2.0, 1.0]], perm2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    a = train.consistency_loss(z1, perm1, z2, perm2, mask, mask)
    b = train.consistency_loss(z2, perm2, z1, perm1, mask, mask)
    assert a.item() > 0.1
    assert abs(a.item() - b.item()) < 1e-6
    # it is the symmetric KL between the two canonical softmaxes
    p1, p2 = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), 0), torch.softmax(torch.tensor([3.0, 2.0, 1.0]), 0)
    ref = 0.5 * ((p1 * (p1.log() - p2.log())).sum() + (p2 * (p2.log() - p1.log())).sum())
    assert abs(a.item() - ref.item()) < 1e-5


def test_consistency_loss_ignores_masked_options_and_different_K():
    # row 0 has 2 options, row 1 has 3; batch 1 is padded to K=4 with garbage in the masked slots,
    # batch 2 is padded to K=3 — the result must equal the unpadded computation (zero here)
    canon = [[1.0, 2.0], [0.0, 1.0, -1.0]]
    perm1, perm2 = [[1, 0], [2, 1, 0]], [[0, 1], [1, 0, 2]]
    z1 = torch.full((2, 4), 50.0)
    z1[0, :2] = torch.tensor([canon[0][j] for j in perm1[0]])
    z1[1, :3] = torch.tensor([canon[1][j] for j in perm1[1]])
    z2 = torch.full((2, 3), -7.0)
    z2[0, :2] = torch.tensor([canon[0][j] for j in perm2[0]])
    z2[1, :3] = torch.tensor([canon[1][j] for j in perm2[1]])
    m1 = torch.tensor([[True, True, False, False], [True, True, True, False]])
    m2 = torch.tensor([[True, True, False], [True, True, True]])
    loss = train.consistency_loss(z1, perm1, z2, perm2, m1, m2)
    assert torch.isfinite(loss) and abs(loss.item()) < 1e-6
    # a real difference confined to the valid options is still seen, and gradients flow to the logits
    z2 = z2.clone().requires_grad_(True)
    with torch.no_grad():
        z2[1, 0] += 2.0
    loss = train.consistency_loss(z1, perm1, z2, perm2, m1, m2)
    assert loss.item() > 0.05
    loss.backward()
    assert torch.isfinite(z2.grad).all() and z2.grad[0, 2].item() == 0.0   # masked slot gets no gradient


def test_reshuffled_batch_gives_same_rows_with_a_different_option_perm():
    rows = P.flatten([_record("r1", "yelp", state="a b c"), _record("r2", "yelp", state="d e")])
    rows_by_id = {r["id"]: r for r in rows}
    tok = FakeTok()
    rng = np.random.default_rng(0)
    (b1,) = list(P.batches(tok, rows, len(rows), shuffle_options=True, rng=rng, device="cpu", shuffle_rows=True))
    b2 = train.reshuffled_batch(tok, b1, rows_by_id, rng, device="cpu", markers=P.QWEN_MARKERS)
    assert b2["id"] == b1["id"] and b2["qtype"] == b1["qtype"] and b2["n_options"] == b1["n_options"]
    # the label follows the options: both batches point at the same canonical option
    for p1, p2, l1, l2 in zip(b1["perm"], b2["perm"], b1["label"].tolist(), b2["label"].tolist()):
        assert p1[l1] == p2[l2]
    choice = [i for i, q in enumerate(b1["qtype"]) if q == "choice"]
    assert any(b1["perm"][i] != b2["perm"][i] for i in choice)
    # noul / score keep their meaningful order in both
    for i, q in enumerate(b1["qtype"]):
        if q != "choice":
            assert b1["perm"][i] == b2["perm"][i] == list(range(b1["n_options"][i]))
    assert b2["ids"].shape[0] == b1["ids"].shape[0] and b2["opt_mask"].shape == b1["opt_mask"].shape
def test_hybrid_head_setup_gives_letter_emb_and_prefix_only_for_hybrid(monkeypatch):
    from tests.test_data import FakeTok
    from tests.test_harness import _FakeTorso
    torso, tok = _FakeTorso(), FakeTok()
    monkeypatch.setattr(P, "LETTER_PREFIX", False)
    monkeypatch.setattr(train.head_mod, "HEAD_CONFIG", {**train.head_mod.HEAD_CONFIG, "name": "hybrid"})
    extra = train.hybrid_head_setup(torso, tok)
    assert extra["letter_emb"].shape == (26, 8) and P.LETTER_PREFIX is True
    heads = [train.head_mod.build_head(8, 4, {**c, **extra}) for c in train.head_configs(4, depth_heads=True)]
    assert len(heads) == 4 and all(type(h).__name__ == "HybridHead" for h in heads)
    monkeypatch.setattr(P, "LETTER_PREFIX", False)
    monkeypatch.setattr(train.head_mod, "HEAD_CONFIG", {**train.head_mod.HEAD_CONFIG, "name": "pointer"})
    assert train.hybrid_head_setup(torso, tok) == {} and P.LETTER_PREFIX is False


def test_micro_batch_override_keeps_effective_batch(monkeypatch):
    monkeypatch.setattr(train, "MICRO_BATCH", 4)
    accum = train.apply_micro_batch(2)
    assert train.MICRO_BATCH == 2 and accum == train.BATCH // 2


def test_option_preview_flag_turns_on_the_module_constant_only_when_set(monkeypatch):
    monkeypatch.setattr(P, "OPTION_PREVIEW", False)
    assert train.apply_option_preview(False) is False and P.OPTION_PREVIEW is False
    assert train.apply_option_preview(True) is True and P.OPTION_PREVIEW is True
