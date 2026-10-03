"""head.py — the EDITABLE read-out for Tier A experiments.

This is the only file the Tier A agent may change. It turns the few hidden-state
vectors that prepare.py cached into one score per option.

Physics reading. The torso is a 24-step discrete-time evolution of n token
vectors in R^d. prepare.py measured a handful of observables from that
evolution: the state at the <answer> marker (which has "seen" the whole
situation and question) and the state at the end of each option (which has
additionally seen that option's text), each taken at several depths
(layers 12, 16, 20, 24). The head is a small function of those observables.
Everything upstream of here is frozen, so training the head costs seconds.

Interface contract with the frozen Tier A trainer (do not change signatures):

    head = build_head(d_model, n_layers)
    logits = head(h_ans, h_opts, opt_mask)
        h_ans    [B, L, d]     state at <answer>, one per tapped layer (L = len(TAP_LAYERS))
        h_opts   [B, L, K, d]  state at each option's terminator, K = max options in batch
        opt_mask [B, K] bool   True where option k exists for that row
        logits   [B, K]        unnormalised score z_k per option; masked entries very negative
    loss = head.loss(logits, labels)      scalar, labels [B] int index of the correct option

The trainer applies softmax(logits / T) with T fitted on dev afterwards, so the
head should not apply its own softmax.

Baseline below: the Strands / Kev pointer head. Two linear maps and a dot product,
    q = W_q h_ans,  k_i = W_k h_i,  z_i = q . k_i / sqrt(width).
It is one attention step whose keys are the options: the question vector points
at the option it matches. With d = 1024 and width = 512 that is 2 * 1024 * 512
= 1,048,576 parameters, no biases.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Edit these to run a variant. "layer" indexes the tapped-layer axis L (−1 = deepest tap).
HEAD_CONFIG = {
    "name": "pointer",
    "width": 512,
    "layer": -1,
}

MASK_VALUE = -1.0e4  # finite so bf16/fp16 and gradient checks stay well-behaved


class PointerHead(nn.Module):
    def __init__(self, d_model: int, width: int, layer: int):
        super().__init__()
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(d_model, width, bias=False)
        self.layer = layer
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, h_ans: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        a = h_ans[:, self.layer]        # [B, d]
        o = h_opts[:, self.layer]       # [B, K, d]
        q = self.q(a)                   # [B, w]
        k = self.k(o)                   # [B, K, w]
        logits = torch.einsum("bw,bkw->bk", q, k) * self.scale
        return logits.masked_fill(~opt_mask, MASK_VALUE)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


def build_head(d_model: int, n_layers: int, cfg: dict | None = None) -> nn.Module:
    """Construct the head the trainer will optimise. n_layers = len(prepare.TAP_LAYERS)."""
    cfg = {**HEAD_CONFIG, **(cfg or {})}
    if cfg["name"] == "pointer":
        return PointerHead(d_model, cfg["width"], cfg["layer"])
    raise ValueError(f"unknown head {cfg['name']!r}")
