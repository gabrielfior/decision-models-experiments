"""CPU tests for the pure parts of train.py."""
import torch

import prepare as P
import train


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
