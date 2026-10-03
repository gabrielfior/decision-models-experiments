# decider-autoresearch

An autoresearch loop that searches **sub-1B Jev-class decision models**: a frozen
language-model torso, the text-generating head deleted, and a small head that scores
developer-supplied options in one forward pass. The loop edits the head (seconds, on
cached torso states) and, less often, the LoRA perturbation of the torso (~25 min on a
24 GB GPU), selects on a 2k-row held-out dev slice with a 2-standard-error gate, and
touches the JevBench public set (231 decisions) only at milestones.

The full plan, with a section that explains every ML term for a physicist, is the
Claude artifact linked in `program.md`. This README is the operational summary.

## Research question

How much of Kev's 0.8B → 4B gap can head architecture, read-out depth and loss
recover without growing the torso?

## Layout

| path | status | role |
|---|---|---|
| `prepare.py` | **frozen** | corpus download, fixed 8k/2k split with two held-out task families, hidden-state cache at layers 12/16/20/24, `score()`, `keep()`, Tier A trainer |
| `head.py` | editable, Tier A | the option-scoring head and its loss (baseline: Kev's pointer head) |
| `train.py` | editable, Tier B | LoRA config, torso, laddered depth heads, optimiser |
| `program.md` | | the agent's instructions and the keep rule |
| `results.tsv` | append-only | one row per experiment |
| `data/splits/` | committed | row ids only; rows are re-downloaded (licence) |
| `compute/` | | Modal app + the $5 budget gate |
| `runpod/` | | pod template, setup script, balance check |
| `jevbench/` | submodule | the benchmark harness, pinned to the commit Strands evaluates against |
| `tests/` | | CPU tests for the scorer, keep rule and head contract |

## Quick start

```bash
uv sync --group dev --group cloud          # local: torch CPU, pytest, modal CLI
uv run pytest                              # 12 tests, CPU, ~5 s
uv run python compute/modal_budget.py      # Modal compute spent this month (cap $5)
uv run modal run compute/modal_app.py::data            # step 2: download + fixed split
uv run modal run compute/modal_app.py::cache           # step 4: hidden-state cache
uv run modal run compute/modal_app.py::train_lora --note "baseline"   # step 3: LoRA baseline
```

Tier A (head-only) runs on a laptop CPU once the cache is pulled from the volume.

## Compute policy

Modal (A10G, serverless, already authenticated) for the first **$5** of compute, then a
RunPod community RTX 3090 (~$0.22/hr) via `runpod/template.json`. Total GPU target for
the programme is ~$7. Stop RunPod pods between sessions.

## Anchors

| model | JevBench public (231) | note |
|---|---|---|
| Strands Decider 2B v19 | 0.723 acc / 0.342 Brier | 167/231 |
| Kev 0.8B | 0.636 acc | Kev's often-quoted 0.697 / 0.397 is on its own locked test set |
| Kev 4B | 0.838 acc / 0.242 Brier (Kev test set) | the gap we are chasing |

## Conventions

- Question types use JevBench's names: `noul` (yes/no), `choice`, `score` (ordinal).
- Brier is the multi-class sum Σ(pᵢ − yᵢ)² in [0, 2], as in JevBench and Strands.
- Dev selection score = mean over types of ½·[(acc − chance)/(1 − chance) + (1 − Brier)].
- Markers reuse Qwen's reserved tokens (Kev's layout); one causal row per question
  because GatedDeltaNet layers carry state across positions.
