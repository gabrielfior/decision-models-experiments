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
import pytest
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


# Scoped to the set-structured heads: the hybrid head reads letter logits by presented SLOT, so it is
# not permutation-equivariant by construction (its own test below checks shapes, masking and the letter path).
@pytest.mark.parametrize("cfg", [{"name": "pointer"}, {"name": "cross_option", "width": 32, "heads": 4, "ffn": 64}])
def test_pointer_and_cross_option_heads_are_equivariant_to_option_permutation(cfg):
    torch.manual_seed(0)
    head = head_mod.build_head(D, L, cfg)
    head.eval()
    h_ans, h_opts, mask, _ = _batch()
    perm = torch.tensor([2, 0, 4, 1, 3])
    base = head(h_ans, h_opts, mask)[1]  # row 1 has all K options
    permuted = head(h_ans, h_opts[:, :, perm], mask[:, perm])[1]
    assert torch.allclose(permuted, base[perm], atol=1e-5)


def _letter_emb(seed, d=D):
    return torch.randn(26, d, generator=torch.Generator().manual_seed(seed))


def test_hybrid_head_keeps_shapes_and_masking_and_reads_the_letter_embeddings():
    h_ans, h_opts, mask, labels = _batch()
    torch.manual_seed(0)
    hybrid = head_mod.build_head(D, L, {"name": "hybrid", "width": 32, "letter_emb": _letter_emb(1)})
    torch.manual_seed(0)
    pointer = head_mod.build_head(D, L, {"name": "pointer", "width": 32})
    torch.manual_seed(0)
    other = head_mod.build_head(D, L, {"name": "hybrid", "width": 32, "letter_emb": _letter_emb(2)})
    logits = hybrid(h_ans, h_opts, mask)
    assert logits.shape == (B, K) and torch.softmax(logits, -1)[0, 3:].max().item() < 1e-6
    assert not torch.allclose(logits[1], other(h_ans, h_opts, mask)[1])          # letter_emb changes the output
    assert hybrid.w.item() == pytest.approx(0.5) and hybrid.w.requires_grad        # learned mixing scalar, init 0.5
    assert not hybrid.letter_emb.requires_grad                                     # frozen torso rows, a buffer
    hybrid.loss(logits, labels).backward()
    assert hybrid.w.grad is not None and hybrid.w.grad.abs().item() > 0
    with torch.no_grad():
        hybrid.w.zero_()
    assert torch.allclose(hybrid(h_ans, h_opts, mask), pointer(h_ans, h_opts, mask), atol=1e-5)   # w = 0 -> plain pointer
    # more than 26 presented options: letter logits exist only for the first 26 slots, the rest is pointer-only
    Kbig = 30
    ho = torch.randn(B, L, Kbig, D)
    mb = torch.ones(B, Kbig, dtype=torch.bool)
    assert hybrid(h_ans, ho, mb).shape == (B, Kbig)
    head_mod.build_head(D, L, {"name": "hybrid", "width": 32, "letter_emb": _letter_emb(1).numpy()})   # numpy accepted
    with pytest.raises(ValueError, match="letter_emb"):
        head_mod.build_head(D, L, {"name": "hybrid", "width": 32})
    assert head_mod.HEAD_CONFIG["name"] == "pointer"                              # default head unchanged


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


def test_cross_option_head_is_selectable_and_keeps_the_contract():
    torch.manual_seed(0)
    head = head_mod.build_head(D, L, {"name": "cross_option", "width": 32, "heads": 4, "ffn": 64, "dropout": 0.1, "residual_pointer": True})
    h_ans, h_opts, mask, labels = _batch()
    head.eval()
    logits = head(h_ans, h_opts, mask)
    assert logits.shape == (B, K) and torch.softmax(logits, -1)[0, 3:].max().item() < 1e-6
    perm = torch.tensor([2, 0, 4, 1, 3])
    assert torch.allclose(head(h_ans, h_opts[:, :, perm], mask[:, perm])[1], logits[1][perm], atol=1e-4)
    assert head_mod.HEAD_CONFIG["name"] == "pointer"          # default head unchanged
    big = head_mod.build_head(2048, 4, {"name": "cross_option"})
    assert sum(p.numel() for p in big.parameters()) < 3_000_000
