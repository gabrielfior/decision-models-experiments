"""Tests for the editable head in head.py, run on CPU with random features.

Interface the frozen Tier A trainer relies on:
    head = head.build_head(d_model, n_layers)
    logits = head(h_ans, h_opts, opt_mask)
      h_ans:    [B, L, d]     hidden state at the <answer> marker, one per tapped layer
      h_opts:   [B, L, K, d]  hidden state at each option's terminator
      opt_mask: [B, K] bool   True where an option exists
      logits:   [B, K]        one score per option; masked options are very negative
    loss = head.loss(logits, labels)   scalar
"""
import torch

import head as head_mod

D, L, K, B = 32, 4, 5, 8


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    h_ans = torch.randn(B, L, D, generator=g)
    h_opts = torch.randn(B, L, K, D, generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 3:] = False  # first row has only 3 options
    labels = torch.randint(0, 3, (B,), generator=g)
    return h_ans, h_opts, mask, labels


def test_logits_have_one_score_per_option_and_mask_missing_options():
    torch.manual_seed(0)
    head = head_mod.build_head(D, L)
    h_ans, h_opts, mask, _ = _batch()
    logits = head(h_ans, h_opts, mask)
    assert logits.shape == (B, K)
    probs = torch.softmax(logits, dim=-1)
    assert probs[0, 3:].max().item() < 1e-6


def test_pointer_head_is_equivariant_to_option_permutation():
    torch.manual_seed(0)
    head = head_mod.build_head(D, L)
    h_ans, h_opts, mask, _ = _batch()
    perm = torch.tensor([2, 0, 4, 1, 3])
    base = head(h_ans, h_opts, mask)[1]  # row 1 has all K options
    permuted = head(h_ans, h_opts[:, :, perm], mask[:, perm])[1]
    assert torch.allclose(permuted, base[perm], atol=1e-5)


def test_loss_decreases_on_a_learnable_toy_problem():
    torch.manual_seed(0)
    head = head_mod.build_head(D, L)
    h_ans, h_opts, mask, labels = _batch()
    # plant the answer: make the correct option's vector equal to the answer vector
    for b in range(B):
        h_opts[b, :, labels[b]] = h_ans[b]
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = head.loss(head(h_ans, h_opts, mask), labels).item()
    for _ in range(50):
        opt.zero_grad()
        loss = head.loss(head(h_ans, h_opts, mask), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_baseline_pointer_head_parameter_count_is_two_projections_plus_norm():
    head = head_mod.build_head(1024, 4)
    n = sum(p.numel() for p in head.parameters())
    width = head_mod.HEAD_CONFIG["width"]
    norm = 2 * 1024 if head_mod.HEAD_CONFIG.get("norm") else 0
    assert n == 2 * 1024 * width + norm  # W_q and W_k (no biases) + LayerNorm affine
