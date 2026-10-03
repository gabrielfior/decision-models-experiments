# program.md — instructions for the autoresearch agent

You are running an autoresearch loop on a decision model: a frozen language-model
torso with a small head that scores developer-supplied options in one forward pass.
The research question, stated once:

> How much of Kev's 0.8B → 4B gap (0.697 → 0.838 accuracy, 0.397 → 0.242 Brier on
> Kev's test set; 0.636 JevBench public at 0.8B) can head architecture, read-out depth
> and loss recover without growing the torso?

Plan: https://claude.ai/artifact/1eaGyQcjcqtkwM5AvLXiNk (read it with the Artifact tool).
Read its section 2 if a term below is unfamiliar. In one sentence each: the torso is a
frozen 24-step evolution of token vectors that we perturb at low rank (LoRA, Tier B)
and tap at an intermediate step (read-out depth, Tier A); the head is a ~0.5M-parameter
function that scores options from a handful of those vectors (Tier A); the loss decides
whether the scores mean likelihoods or ranks (Tier A); and the temperature rescales the
scores into honest probabilities afterwards.

## Files

| file          | who edits it      | what it holds                                                        |
|---------------|-------------------|----------------------------------------------------------------------|
| `prepare.py`  | nobody (frozen)   | data subset, fixed split, feature cache, `score()`, `keep()`, Tier A trainer |
| `head.py`     | Tier A agent      | the read-out module and its loss; see its docstring for the contract |
| `train.py`    | Tier B agent      | LoRA config, torso choice, depth heads, optimiser                    |
| `results.tsv` | append-only       | one row per experiment                                               |
| `jevbench/`   | nobody            | the benchmark harness, run only at milestones, never inside the loop |

## Tiers

**Tier A** (seconds, no torso forward pass; runs on a laptop CPU). Edit `head.py` only. Run

    uv run python prepare.py tier-a --note "<one line: what you changed and why>"

It trains your head for `N_SEEDS` seeds on the cached hidden states, scores the dev
slice, appends a row to `results.tsv`, and prints KEEP or DISCARD.

**Tier B** (20–25 min on a 24 GB GPU). Edit `train.py` only. Run

    uv run modal run compute/modal_app.py::train_lora --note "<one line>"     # first $5
    uv run python train.py --note "<one line>"                                # on the RunPod pod

Promote at most the top three Tier A results per day to Tier B.

## The keep rule (enforced by prepare.keep, do not reimplement)

- Dev score = mean over question types (noul, choice, score) of
  0.5 × ( (accuracy − chance)/(1 − chance) + (1 − Brier) ).
  Brier is the multi-class sum Σ(p_i − y_i)², the same convention as JevBench.
- A change is KEPT only if its 3-seed mean dev score beats the current baseline's
  3-seed mean by MORE than two noise floors, where the noise floor is the larger of
  the dev-set standard error (`selection_se`, ~0.011 at 2k rows) and the seed standard
  error (std over seeds / √3). The first pointer baseline had a seed spread three times
  the dev SE, so the seed term is what usually binds. Otherwise `git checkout -- head.py`
  (or `train.py`) and move on.
- Why two standard errors: Strands v17 → v19 moved JevBench by three decisions out
  of 231, and their hard-tier score did not move at all. A loop selecting on a few
  hundred items chases noise within an afternoon. 2k dev rows × 3 seeds × a 2-SE
  gate is the minimum that makes a kept change mean something.
- ECE and p50 latency are reported, not selected on.
- Option order is shuffled at training time and scored under two orders at eval.
  Order sensitivity above 2 percentage points is a failure regardless of accuracy.
- Held-out task families are reported next to dev and never used to select.
- JevBench public (231 decisions) is run at the start, after each Tier B promotion,
  and at the end. Never inside the loop.

## Rules

1. One idea per experiment. Say the hypothesis in `--note` before you run it.
2. Never edit `prepare.py`, the split ids under `data/splits/`, or `jevbench/`.
3. Never read dev or held-out labels in `head.py`/`train.py`. The trainer gives you
   train labels only.
4. Every run appends to `results.tsv` with the commit hash. Commit kept changes with
   the note as the message. Do not amend history.
5. If the slot head (negative control) is not separated from the pointer head by the
   gate, stop: the dev set is too small and every later result is suspect.
6. Money: Modal compute is capped at $5/month (`compute/modal_budget.py`); after that,
   Tier B runs on RunPod. Stop RunPod pods when a batch finishes.

## Queue for the first Tier A night (in order of expected gain)

1. Multi-layer read-out: concatenate or softmax-mix the four tapped layers.
2. Cross-option attention: one small transformer layer over {answer, options} before scoring.
3. Loss variants: CE + Brier, label smoothing ε = 0.05 / 0.1.
4. Ordinal (cumulative-link) output for `score` questions.
5. Confidence / abstain head (second scalar output; must not change the ranking).
6. Hashed n-gram side channel concatenated to the answer vector.
7. Controls: slot head (negative), letter-logit read-out (zero-parameter).

## results.tsv columns

    timestamp  commit  tier  note  seeds  dev_selection  dev_se  dev_acc  dev_brier  dev_ece  heldout_acc  heldout_brier  order_sens  p50_ms  kept  per_seed  baseline
