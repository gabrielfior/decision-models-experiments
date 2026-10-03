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
