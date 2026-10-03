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
at the option it matches. Kev's 0.8B head uses width = 256; with d = 1024 that is
2 * 1024 * 256 = 524,288 parameters here (Kev adds biases, 524,800; a bias on k
adds the same constant to every option and cancels in the softmax, so it is left
out). Strands puts a LayerNorm in front and uses d = 2048, giving ~1.05M.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Edit these to run a variant. "layer" indexes the tapped-layer axis L (−1 = deepest tap).
HEAD_CONFIG = {
    "name": "pointer",            # "pointer" (Kev/Strands dot-product), "cross_option" (options attend to each other first)
                                  # or "hybrid" (pointer + w * the torso's own letter logits; needs cfg["letter_emb"] and LETTER_PREFIX)
    "width": 256,
    "layer": 1,                   # index into the tapped-layer axis: 0/1/2/3 = 50/67/83/100% depth
    "norm": True,                 # LayerNorm before q/k: shallow taps (norm 2-6) and deep taps (norm ~120) on one scale
    # cross_option only
    "heads": 4,
    "ffn": 512,
    "dropout": 0.1,
    "residual_pointer": True,     # add the plain pointer(LN(h)) score so attention only learns a correction
}

MASK_VALUE = -1.0e4  # finite so bf16/fp16 and gradient checks stay well-behaved


class PointerHead(nn.Module):
    def __init__(self, d_model: int, width: int, layer: int, norm: bool = False):
        super().__init__()
        # One LayerNorm shared by the answer and option vectors: both come from the same
        # tapped layer, so one scale/shift suffices and option equivariance is preserved.
        self.norm = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(d_model, width, bias=False)
        self.layer = layer
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, h_ans: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        a = self.norm(h_ans[:, self.layer])        # [B, d]
        o = self.norm(h_opts[:, self.layer])       # [B, K, d]
        q = self.q(a)                   # [B, w]
        k = self.k(o)                   # [B, K, w]
        logits = torch.einsum("bw,bkw->bk", q, k) * self.scale
        return logits.masked_fill(~opt_mask, MASK_VALUE)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class CrossOptionHead(nn.Module):
    """Pointer head whose answer and option vectors first talk to each other.

    LN the tapped vectors, project answer and options to `width`, run ONE pre-LN
    transformer encoder layer over the set {ans, opt_1..K} (no positional encoding,
    padded options masked out as keys), then score z_k = q(ans') . k(opt_k') / sqrt(width).
    Self-attention over an unordered set is permutation-equivariant, so option
    equivariance is preserved. Optionally add the plain pointer score on LN(h) as a
    residual so the attention layer only has to learn a correction.
    """

    def __init__(self, d_model: int, width: int, layer: int, norm: bool = True, heads: int = 4,
                 ffn: int = 512, dropout: float = 0.1, residual_pointer: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.proj_a = nn.Linear(d_model, width)
        self.proj_o = nn.Linear(d_model, width)
        self.enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.out_norm = nn.LayerNorm(width)  # pre-LN blocks leave the residual stream un-normalised
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.residual_pointer = residual_pointer
        if residual_pointer:
            self.q0 = nn.Linear(d_model, width, bias=False)
            self.k0 = nn.Linear(d_model, width, bias=False)
        self.layer = layer
        self.scale = 1.0 / math.sqrt(width)
        # Start in inference mode: dropout makes two train-mode forward passes differ, which is
        # noise rather than a lack of equivariance. The Tier A trainer calls .train() before each
        # epoch and .eval() before scoring, so this changes nothing about how the head is trained.
        self.eval()

    def forward(self, h_ans: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        a = self.norm(h_ans[:, self.layer])        # [B, d]
        o = self.norm(h_opts[:, self.layer])       # [B, K, d]
        x = torch.cat([self.proj_a(a).unsqueeze(1), self.proj_o(o)], dim=1)   # [B, 1+K, w]
        pad = torch.cat([torch.zeros_like(opt_mask[:, :1]), ~opt_mask], dim=1)  # True = ignore
        y = self.out_norm(self.enc(x, src_key_padding_mask=pad))
        q = self.q(y[:, 0])                         # [B, w]
        k = self.k(y[:, 1:])                        # [B, K, w]
        logits = torch.einsum("bw,bkw->bk", q, k) * self.scale
        if self.residual_pointer:
            logits = logits + torch.einsum("bw,bkw->bk", self.q0(a), self.k0(o)) * self.scale
        return logits.masked_fill(~opt_mask, MASK_VALUE)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class HybridHead(nn.Module):
    """Pointer score plus the torso's own next-token logit for the option's slot letter (Mapika / Intern-Decision-2B).

    With options presented as "A) ...", "B) ...", the base model's LM head at <decide> already ranks the
    letters; with tied embeddings that logit is h_last · E[" A"], so no LM head has to be kept, only the
    26 letter rows of the embedding table (`letter_emb`, a frozen buffer). The read-out is
        z_j = pointer(h_ans, h_opts)_j + w * LN(h_ans[:, -1]) · letter_emb[j]        for slot j < 26,
    with w a learned scalar (init 0.5) and a LayerNorm on the letter path's input. h_ans[:, -1] is the
    deepest tap: in this repo the last layer's post-norm state, i.e. what the LM head would read.
    Letter logits are indexed by PRESENTED slot, so this head is not permutation-equivariant by
    construction; slots beyond the 26 letters (banking77) get the pointer score alone.
    """

    def __init__(self, d_model: int, width: int, layer: int, norm: bool, letter_emb):
        super().__init__()
        self.pointer = PointerHead(d_model, width, layer, norm)
        emb = torch.as_tensor(letter_emb).detach().clone().float()
        if emb.ndim != 2 or emb.shape[1] != d_model:
            raise ValueError(f"letter_emb must be [n_letters, {d_model}], got {tuple(emb.shape)}")
        self.register_buffer("letter_emb", emb)                # [26, d], frozen torso rows
        self.letter_norm = nn.LayerNorm(d_model)
        self.w = nn.Parameter(torch.tensor(0.5))

    def letter_logits(self, h_ans: torch.Tensor, K: int) -> torch.Tensor:
        last = self.letter_norm(h_ans[:, -1])                   # [B, d]
        n = min(K, self.letter_emb.shape[0])
        ll = last @ self.letter_emb[:n].t()                     # [B, n]
        if n < K:
            ll = F.pad(ll, (0, K - n))                          # no letter for slots >= 26: contribute 0
        return ll

    def forward(self, h_ans: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        z = self.pointer(h_ans, h_opts, opt_mask) + self.w * self.letter_logits(h_ans, opt_mask.shape[1])
        return z.masked_fill(~opt_mask, MASK_VALUE)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


def build_head(d_model: int, n_layers: int, cfg: dict | None = None) -> nn.Module:
    """Construct the head the trainer will optimise. n_layers = len(prepare.TAP_LAYERS).
    For name == "hybrid", cfg["letter_emb"] ([26, d] tensor or array from the cache / torso) is required."""
    cfg = {**HEAD_CONFIG, **(cfg or {})}
    if cfg["name"] == "pointer":
        return PointerHead(d_model, cfg["width"], cfg["layer"], cfg.get("norm", False))
    if cfg["name"] == "cross_option":
        return CrossOptionHead(d_model, cfg["width"], cfg["layer"], cfg.get("norm", True),
                               cfg.get("heads", 4), cfg.get("ffn", 512), cfg.get("dropout", 0.1),
                               cfg.get("residual_pointer", False))
    if cfg["name"] == "hybrid":
        if cfg.get("letter_emb") is None:
            raise ValueError("head 'hybrid' needs cfg['letter_emb'] ([26, d] letter embeddings from the cache or torso)")
        return HybridHead(d_model, cfg["width"], cfg["layer"], cfg.get("norm", True), cfg["letter_emb"])
    raise ValueError(f"unknown head {cfg['name']!r}")
