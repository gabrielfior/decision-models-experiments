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

## 2026-10-03/04 — goal 2: beat Strands Decider 2B (167/231) with our own architecture

Everything at 2B + LoRA sits in a 161–167 band on JevBench public regardless of head (plain pointer, LayerNorm@16,
cross-option attention, hybrid letter-logit read-out, half-depth read-out at 146), and seed noise is about ±4
decisions. Averaging predictions over option orders (two orders: +5–6 decisions; 4 or 8 orders: no more) and over
models (ensembles of 2–3) raises the floor to 164–167 and saturates there. LoRA weight soups across seeds hurt
(155–164). Distillation from an external decision model was excluded by decision: the result must be ours.

What broke the band was training data, not the head: training on train + the two held-out families (9.9k rows
instead of 8k, temperatures fitted on dev) with the LayerNorm@16 head and two epochs gives dev acc 0.768/0.769 for
seeds 0/1, JevBench 166/166 plain and **168/168 with two-order averaging**, and the two-seed ensemble with
two-order averaging reaches **170/231 = 0.736 (Brier 0.367)**, three decisions above Strands' 167 (Brier 0.342).
Each piece is this repo's own: the token layout and LayerNorm pointer head, the LoRA recipe (Kev's targets), the
two-order inference, and the data split. Still open: the Brier gap to Strands, and the abstain-augmentation,
option-preview and shuffle-consistency variants (scored at the end of the session).

**Goal 2, closing the remaining lanes.** Shuffle-consistency loss (two option orders per step, symmetric KL, R-Drop):
dev acc 0.764, order sensitivity 0.8% (lowest of any run), JevBench 165 / 165 / 164 with 1 / 2 / 4 orders: it makes the
model order-invariant in training, so test-time averaging stops helping, but accuracy stays in the band. Abstain /
unrelated-label augmentation: dev 0.765, JevBench 164 / 165. Option preview before the state: dev 0.756, JevBench
161 / 163 (worse). Three-model ensemble (two data-lever seeds + augmentation): 170, same as the two-seed ensemble.
Final: **170/231 (0.736), Brier 0.367**, our own recipe, vs Strands Decider 2B 167/231 (0.723), Brier 0.342.

**Framing correction (2026-10-04).** The 168/168/170 results differ from the 165–167 band in ONE respect: the
training set. They were trained on train + the two task families the plan had held out (trec, legacy_policy), 9.9k
rows instead of 8k. On the original 8k rows every architecture variant (LayerNorm head, cross-option attention,
hybrid letter logits, consistency loss, augmentation, option preview, ensembles, order averaging) tops out at
165–167 = Strands' 167. So the honest claim is: **our recipe ties Strands Decider 2B on 8k rows and passes it with
24% more training data.** The extra decisions are a data contribution; and the 9.9k models have no held-out task
family left, so the 231 public JevBench items are their only generalisation check.

**Architecture-only ensemble (8k rows, no extra data).** Averaging four of our 8k-row variants (LayerNorm@16 seed 1,
abstain augmentation, hybrid letter-logit read-out, shuffle-consistency loss) with two-order averaging: **168/231 =
0.727, Brier 0.370** — one decision over Strands' 167 with the same 8k training rows, inside one standard error.
A two-member version scores 166. The third 9.9k seed was stopped at step 100 to stay under the $20 Modal line.

**Extension (2026-10-04, +$10 Modal).** Third seeds: 9.9k recipe 168 plain / 171 with two-order averaging (three seeds
with averaging: 168, 168, 171; three-seed ensemble 170 plain / **171 averaged, Brier 0.364**); 8k recipe 163 / 160 (three
seeds with averaging: 167, 165, 160; four-member 8k-only ensembles 166–168). Consistency loss trained on the 9.9k rows:
dev acc 0.770, order sensitivity 0.75%, JevBench 166 plain / 168 averaged, i.e. the same as the plain recipe. Final
statement for the post: on 8k rows the recipe ties Strands (167); on 9.9k rows every seed passes it (168–171) and the
three-seed ensemble reaches 171/231 = 0.740. Modal total $25.3, RunPod $3.3.

**Session 3 (2026-10-04 evening): newer base torsos, and a publishable model under 2B.** Budget $3 Modal + $3 RunPod,
unlimited Mac time (M3 Pro, 36 GB: the 2B decider runs at ~0.5 s/row on the Mac GPU, so dev evaluation and JevBench
scoring moved local; GPUs only train).

*Newer Qwen.* There is none to try: every Qwen release after Qwen3.5 (3.6, 3.7, 3.8) ships at 27B or as a large MoE and
without a base checkpoint (HF org listing checked 2026-10-04). The newest Qwen base weights at <= 2B remain
Qwen3.5-2B-Base / 0.8B-Base (Feb 2026, weights unchanged). A community Qwen3.8 -> 2B distillation (instruct) reached
dev acc 0.774 / Brier 0.298 in one run and 161 / 163 on JevBench (inside the Qwen3.5-2B band; kept under runs/excluded,
not in the tables), but distilled and instruct torsos were excluded by decision: regular base models only. So the lane became "newest regular base models <= 2B from any lab", same recipe (LayerNorm head at
~67% depth + LoRA r16, 2 epochs, 8k rows), each scored plain and with two-order averaging:

| torso (release) | params | dev acc / Brier | held-out acc | JevBench plain / 2-order | Brier |
|---|---|---|---|---|---|
| Qwen3.5-2B-Base (Feb 2026), seeds 0/1/2 | 1.88B text | 0.757–0.774 / 0.31 | 0.69–0.70 | 162–164 / 160–167 | 0.37–0.39 |
| MiniCPM5-2B-Base (Aug 2026), 42 L, tap 28 | 2.52B | 0.765 / 0.307 | 0.730 | 152 / 155 | 0.405 |
| MiniCPM5-1B-Base (May 2026), 24 L, tap 16 | 1.08B | 0.706 / 0.366 | 0.509 | 134 / 135 | 0.493 |
| LFM2.5-2.6B-Base (Aug 2026), 30 L, tap 20 | 2.70B | 0.7625 / 0.314 | 0.681 | 158 / 157 | 0.429 |
| granite-swash-2b (Jul 2026), 24 L, tap 16 | 2.14B | GRANITE_DEV | | GRANITE_JEV | |
| Qwen3.5-0.8B-Base, 8k (session 1) | 0.87B | 0.741 | | 152 | |

MiniCPM5-2B is the instructive case: better in-distribution dev numbers than Qwen3.5-2B and +3 points on the held-out
decision-v7 families, yet 10 decisions worse on JevBench. The JevBench items are the only out-of-corpus check we have,
and on them the Qwen3.5 torso generalises best. OTHER_TORSO_SENTENCE

*Cactus Compute.* Their own models (needle 1/2/3, 26–121M) are from-scratch attention-only function callers trained on
100–360B synthetic tokens with 2-bit quantisation-aware training; none of that grafts onto a pretrained torso, and on
public suites Qwen3.5-0.8B beats them. What does transfer: (1) a learned scorer on an intermediate layer's hidden
state (their Gemma-4 confidence probe reads layer 28 of 35) is the same object as our pointer head; (2) cut the model
at the layer you read; (3) keep embeddings at higher precision than the body when quantising; (4) their 8k-token
vocabulary is the extreme form of "the embedding table is dead weight for a scorer".

*The published model can be 1.12B with the same decisions.* The head reads layer 16 of 24, so layers 17–24 never
influence a decision and received no gradient under LoRA: deleting them is exact (max |Δlogit| = 0 on dev; checked
on the Mac). 1.88B -> 1.42B. Then the vocabulary: Qwen's byte-level BPE assigns ids in merge order, and ids >= 98,304
carry 0.15% of our corpus tokens and 0.21% of JevBench's; keeping the first 98,304 pieces (+ the special tokens) and
re-splitting the rest with the remaining merges shrinks the embedding table from 509M to 201M parameters. Exported
9.9k seed 0, full dev split (2000 rows): original 1.88B acc 0.7675 / Brier 0.3112 / selection 0.675; exported 1.12B
acc 0.769 / 0.3113 / 0.676; JevBench 165 / 167 (two-order) vs 166 / 168 for the original — one decision, inside
noise. Weight-only quantisation of the 16-layer model on dev: INT8 per-channel acc 0.771 / Brier 0.311 (free);
INT4 group-128 acc 0.7645 / Brier 0.322 (−0.4 points, +0.01 Brier). `scripts/export.py` writes the artefact
(merged LoRA, 16 layers, trimmed tokenizer, head, temperatures); `serve.py --torso <export>/torso --run <export>` serves it.
SMALL_QWEN_SENTENCE
