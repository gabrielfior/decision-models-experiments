# Findings log

One entry per result that changed what we believe. Numbers are the dev "selection" score
(mean over question types of ½[(acc − chance)/(1 − chance) + (1 − Brier)]) unless stated.
Full rows: `results.tsv` (Qwen), `results-lfm.tsv`, `results-modernbert.tsv`, `docs/jevbench-public.tsv`.

## 2026-10-03 — session 1

**Setup.** decision-v7 (Kev): 8,000 train / 2,000 dev (stratified by type, split by Kev's
group_id) / 1,896 held-out (trec + legacy_policy). Hidden states cached at layers 12/16/20/24
(Qwen3.5-0.8B), 7/9/12/14 (LFM2.5-230M), 14/19/23/28 (ModernBERT-large). Tier A trains a head on the
cache (laptop CPU, ~30 s for 3 seeds). Tier B trains LoRA r16 + head for 1 epoch (12 min on an A10G,
10.5 min on a 3090).

**The plain pointer baseline was a position prior.** Kev-shape pointer head on frozen layer-24
states: 0.287. A head that never sees the options: 0.285. Cause: layer-24 states have norm ~120, so
q·k/√256 saturates the softmax on step one. One LayerNorm on the inputs fixes it (0.399 at layer 24).

**Read-out depth matters a lot on a frozen torso.** LayerNorm pointer by depth, Qwen:
12 → 0.408, **16 → 0.437**, 20 → 0.388, 24 → 0.399. Replicated on fresh seeds (0.441). Same shape on
LFM2.5 (best at layer 9 of 14: 0.302 vs 0.279 at the top) and ModernBERT (strictly monotone, best at
layer 14 of 28: 0.286 vs 0.242 at the top). Dev accuracy 0.463 → 0.578 on Qwen.

**Nothing else moved the frozen-torso head beyond the gate** (2 × max(dev SE, seed SE) ≈ 0.023):
multi-layer mixes (0.432–0.436), cross-option attention (0.377 at layer 24; 0.451 with a residual
pointer at layer 16, +0.014), width 1024 (0.440), CE+Brier (does not stack), label smoothing (hurts),
input dropout (hurts). Position-only control vs the layer-16 head: 15 noise floors apart.

**With LoRA, the head stops mattering for accuracy.** Qwen3.5-0.8B + LoRA r16, 8k rows, 1 epoch:
plain head seeds 0/1 → 0.629 / 0.620 (acc 0.732 / 0.723); LayerNorm@16 head → 0.641 / 0.619
(acc 0.743 / 0.722). Δ of means +0.006 vs seed std ~0.012. Order sensitivity is lower with the
LayerNorm head in both seeds (2.2% / 3.6% vs 4.6% / 5.3%). Adapting the torso doubles what the
frozen torso gives; this is the torso-capacity view of the 0.8B→4B gap.

**JevBench public (231).** Kev-0.8B published 0.636. Ours, plain head + LoRA: 145/231 = 0.628
(easy 48/48, standard 54/72, hard 43/111). LayerNorm@16 head + LoRA: 152/231 = 0.658
(47/48, 55/72, 50/111). Frozen torso + LayerNorm@16 head, no LoRA: 112/231 = 0.485
(42/48, 31/72, 39/111). Strands-2B published 0.723. Differences of ~7 decisions are one binomial SE.

**Harness lessons.** The gate's noise floor must include the seed SE (first baseline's seed spread was
3× the dev SE). Lanes must gate against a pinned baseline, not their own last keep. LoRA dropout and
gradient checkpointing were silently off until `torso.train()` was added (all Tier B rows above trained
with dropout 0). JevBench temperatures fitted on our dev slice over-sharpen the hard tier (ECE 0.09–0.19).

**Torso swap with LoRA (1 epoch, 8k rows, LayerNorm pointer head at tap 1).**

| torso | params | dev acc | dev sel | held-out acc | order sens | JevBench | CPU p50 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 752M text | 0.743 | 0.641 | 0.605 | 2.2% | 152/231 = 0.658 | 2.3 s |
| LFM2.5-230M-Base | 230M | 0.632 | 0.508 | 0.430 | 4.1% | 110/231 = 0.476 | 0.23 s |
| ModernBERT-large | 395M | 0.608 | 0.477 | 0.382 | 7.6% | 106/231 = 0.459 | 0.14 s |

The 0.8B torso loses ~0.1 accuracy from dev to JevBench; the two small torsos lose ~0.15, and their
held-out-family accuracy is much lower: the smaller models memorise the training families and generalise
less. ModernBERT's order sensitivity is the highest of all, which fits a bidirectional model that sees
every option while reading each one. Neither small torso approaches Kev-0.8B (0.636) with this recipe;
the 0.8B remains the smallest torso in this ladder that is competitive. LoRA rank 64 on Qwen did not beat
rank 16 (0.6135 vs 0.6195 selection, held-out 0.582 vs 0.611).

**Epochs.** LayerNorm@16 head + LoRA r16, Qwen3.5-0.8B, 8k rows: 1 epoch → dev 0.641 / held-out 0.605 /
order sens 2.2% / JevBench 152; 2 epochs → dev 0.641 / held-out 0.658 / order sens 1.4% / JevBench 152
(Brier 0.446 → 0.446, ECE 0.094 → 0.087). The second epoch improves generalisation to held-out task
families and stability, not JevBench accuracy. Laddered depth heads (0.626) and LoRA rank 64 (0.614) did not
beat the single head at rank 16 (0.620–0.641 across seeds).

**Scaling check (plan step 8).** Qwen3.5-2B-Base + LoRA r16 + LayerNorm@16 head, 1 epoch, 8k rows, 23 min on a 3090:
dev acc 0.761 / sel 0.666 / Brier 0.316 / held-out 0.724 / order sens 1.05%; JevBench 162/231 = 0.701. The
0.8B with the same recipe: 0.743 / 0.641 / 0.344 / 0.605 / 2.2%; JevBench 152. The torso step is worth +0.12
held-out accuracy and +10 JevBench decisions; Strands' 2B (0.723, 123k rows) is 5 decisions away.

**Three epochs** (0.8B, LayerNorm@16 head, LoRA r16): dev 0.629 / held-out 0.638 / order sens 1.25%, no better
than two epochs within seed noise. Final-model candidate for this recipe: 0.8B, LayerNorm@16 head, 2 epochs
(JevBench 152/231); the 2B at 1 epoch reaches 162/231.
