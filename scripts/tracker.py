"""Render the autoresearch progress tracker (tracker.html) from the results files.

Reads results.tsv, results-<torso>.tsv, every workflow worktree's results*.tsv and docs/jevbench-public.tsv,
dedupes experiments, and draws inline SVG: dev selection per experiment in run order with the running best,
one panel per torso for Tier A, one for Tier B, and JevBench accuracy per scored model.
    uv run python scripts/tracker.py [out.html]
"""
from __future__ import annotations

import csv
import glob
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "tracker.html"

# ---------------------------------------------------------------- data
def load_rows():
    files = ["results.tsv", "results-lfm.tsv", "results-modernbert.tsv", "results-qwen2b.tsv"] + sorted(glob.glob(str(ROOT / ".claude/worktrees/*/results*.tsv")))
    seen, rows = set(), []
    for f in files:
        p = ROOT / f if not str(f).startswith("/") else Path(f)
        if not p.exists():
            continue
        torso = "LFM2.5-230M" if "lfm" in p.name else "ModernBERT-large" if "modernbert" in p.name else "Qwen3.5-2B" if "qwen2b" in p.name else "Qwen3.5-0.8B"
        for r in csv.DictReader(open(p), delimiter="\t"):
            key = (r["note"], r.get("per_seed", ""), r["tier"])
            if key in seen or r["note"].startswith("smoke"):
                continue
            seen.add(key)
            try:
                sel = float(r["dev_selection"])
            except ValueError:
                continue
            n = r["note"]
            if r["tier"] == "B":
                torso = "LFM2.5-230M" if "LFM" in n else "ModernBERT-large" if "ModernBERT" in n else "Qwen3.5-2B" if "2B" in n else "Qwen3.5-0.8B"
            rows.append({"ts": r["timestamp"], "tier": r["tier"], "torso": torso, "note": n, "sel": sel,
                         "acc": float(r["dev_acc"] or 0), "brier": float(r["dev_brier"] or 0), "held": float(r["heldout_acc"] or 0),
                         "kept": r["kept"] in ("True", "true"), "src": p.name})
    # chronological; worktree timestamps are ISO, Modal rebuilt rows are MMDDHHMM — normalise to sortable
    def k(r):
        t = r["ts"]
        return t if "T" in t else f"2026-{t[:2]}-{t[2:4]}T{t[4:6]}:{t[6:8]}:00"
    rows.sort(key=k)
    return rows


def load_jevbench():
    p = ROOT / "docs" / "jevbench-public.tsv"
    out = []
    if p.exists():
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r["run"].startswith("smoke"):
                continue
            out.append({"run": r["run"], "acc": float(r["accuracy"]), "n": int(r["n_correct"]), "hard": r["hard"], "brier": float(r["brier"])})
    return out


# ---------------------------------------------------------------- svg helpers
W, H, PAD = 760, 300, dict(l=56, r=16, t=28, b=44)
LABELS = {"tierB-baseline-plain-pointer": "0.8B · plain head + LoRA", "tierB-ln16-head": "0.8B · LN@16 head + LoRA",
          "tierB-ln16-head-2epochs": "0.8B · LN@16 + LoRA, 2 epochs", "tierA-frozen-ln16-head": "0.8B frozen · LN@16 head",
          "tierB-lfm230m-lora": "LFM2.5-230M + LoRA", "tierB-modernbert-lora": "ModernBERT-large + LoRA",
          "tierB-qwen2b-ln16": "2B · LN@16 head + LoRA", "tierB-2b-final-10k-2ep": "2B · LN@16 + LoRA, 10k rows, 2 epochs",
          "tierB-2b-final-10k-2ep-tta": "2B · 10k rows, 2 epochs, two-order TTA",
          "tierB-2b-ln16-2ep": "2B · LN@16 + LoRA, 8k rows, 2 epochs", "tierB-2b-ln16-2ep-tta": "2B · 8k, 2 epochs, TTA2",
          "tierB-2b-xopt-1ep": "2B · cross-option head + LoRA", "tierB-2b-xopt-1ep-tta": "2B · cross-option head, TTA2",
          "tierB-2b-ln16-s1": "2B · LN@16 + LoRA, seed 1", "tierB-2b-ln16-s1-tta": "2B · seed 1, TTA2",
          "tierB-2b-half-depth": "2B · read layer 12 (half depth)", "tierB-2b-half-depth-tta": "2B · half depth, TTA2",
          "2ep8k-tta4": "2B · 8k, 2 epochs, TTA4", "2ep8k-tta8": "2B · 8k, 2 epochs, TTA8", "10k2ep-tta4": "2B · 10k, 2 epochs, TTA4", "10k2ep-tta8": "2B · 10k, 2 epochs, TTA8",
          "tierB-2b-hybrid-2ep": "2B · hybrid read-out (pointer + letter logits)", "tierB-2b-hybrid-2ep-tta2": "2B · hybrid, TTA2", "tierB-2b-hybrid-2ep-tta4": "2B · hybrid, TTA4",
          "soup-8k-s01": "2B LoRA soup (8k seeds 0+1)", "soup-8k-s01-tta2": "2B soup 8k, TTA2", "soup-10k-s01": "2B LoRA soup (10k seeds 0+1)", "soup-10k-s01-tta2": "2B soup 10k, TTA2",
          "soup-all4": "2B LoRA soup (all 4)", "soup-all4-tta2": "2B soup all 4, TTA2", "2ep8k-retemp": "2B · 8k 2 epochs, temps refit", "2ep8k-retemp-tta2": "2B · 8k 2 epochs, temps refit, TTA2",
          "2ep8k-s1": "2B · 8k, 2 epochs, seed 1", "2ep8k-s1-tta2": "2B · 8k, 2 epochs, seed 1, TTA2", "10k2ep-s1": "2B · 10k, 2 epochs, seed 1", "10k2ep-s1-tta2": "2B · 10k, 2 epochs, seed 1, TTA2",
          "9k9-2ep-s1": "2B · 9.9k rows (train+held-out), 2 epochs, seed 1", "9k9-2ep-s1-tta2": "2B · 9.9k rows, seed 1, TTA2",
          "9k9-2ep-s0": "2B · 9.9k rows (train+held-out), 2 epochs, seed 0", "9k9-2ep-s0-tta2": "2B · 9.9k rows, seed 0, TTA2",
          "ens-9k9-s01-tta2": "2B · 9.9k rows, ensemble of seeds 0+1, TTA2",
          "augment-2ep": "2B · abstain/unrelated augmentation, 2 epochs", "augment-2ep-tta2": "2B · augmentation, TTA2",
          "preview-2ep": "2B · option preview before state, 2 epochs", "preview-2ep-tta2": "2B · option preview, TTA2",
          "ens-9k9+aug-tta2": "2B · 9.9k seeds 0+1 + augmentation, ensemble of 3, TTA2",
          "tierB-2b-consistency": "2B · shuffle-consistency loss (R-Drop), 2 epochs", "tierB-2b-consistency-tta2": "2B · consistency loss, TTA2", "tierB-2b-consistency-tta4": "2B · consistency loss, TTA4",
          "ens-8k-only-s1+aug-tta2": "2B · 8k rows only: ensemble of 2 variants, TTA2", "ens-8k-only-4-tta2": "2B · 8k rows only: ensemble of 4 architecture variants, TTA2",
          "9k9-2ep-s2": "2B · 9.9k rows, seed 2", "9k9-2ep-s2-tta2": "2B · 9.9k rows, seed 2, TTA2", "2ep8k-s2": "2B · 8k, 2 epochs, seed 2", "2ep8k-s2-tta2": "2B · 8k, 2 epochs, seed 2, TTA2",
          "ens-9k9-s012-tta2": "2B · 9.9k rows, ensemble of 3 seeds, TTA2", "ens-9k9-s012": "2B · 9.9k rows, ensemble of 3 seeds", "ens-8k-only-s12+cons+aug-tta2": "2B · 8k only: seeds 1+2 + consistency + augmentation, TTA2",
          "9k9-consistency": "2B · consistency loss on 9.9k rows", "9k9-consistency-tta2": "2B · consistency on 9.9k, TTA2",
          "ens-2ep8k+10k-tta2": "2B ensemble of 2, TTA2", "ens-2ep8k+10k-tta4": "2B ensemble of 2, TTA4", "ens-3-tta2": "2B ensemble of 3, TTA2", "ens-3-tta4": "2B ensemble of 3, TTA4"}


def sx(i, n):
    return PAD["l"] + (W - PAD["l"] - PAD["r"]) * (i + 0.5) / max(n, 1)


def sy(v, lo, hi):
    return PAD["t"] + (H - PAD["t"] - PAD["b"]) * (1 - (v - lo) / (hi - lo))


def panel(rows, title, lo, hi, show_legend):
    n = len(rows)
    if n == 0:
        return f'<p class="muted">{html.escape(title)}: no experiments yet.</p>'
    g = []
    ticks = [lo + (hi - lo) * i / 4 for i in range(5)]
    for t in ticks:
        y = sy(t, lo, hi)
        g.append(f'<line x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        g.append(f'<text x="{PAD["l"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>')
    # running best over kept experiments (step line)
    best, pts = None, []
    for i, r in enumerate(rows):
        if r["kept"] and (best is None or r["sel"] > best):
            if best is not None:
                pts.append((sx(i, n), sy(best, lo, hi)))
            best = r["sel"]
        if best is not None:
            pts.append((sx(i, n), sy(best, lo, hi)))
    if pts:
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        g.append(f'<path d="{d}" class="best"/>')
        bx, by = pts[-1]
        g.append(f'<text x="{min(bx + 8, W - PAD["r"] - 70):.1f}" y="{by - 8:.1f}" class="lbl">best {best:.3f}</text>')
    for i, r in enumerate(rows):
        x, y = sx(i, n), sy(r["sel"], lo, hi)
        cls = "kept" if r["kept"] else "disc"
        tip = html.escape(f'{r["note"]}\nselection {r["sel"]:.4f} · acc {r["acc"]:.3f} · Brier {r["brier"]:.3f} · held-out acc {r["held"]:.3f} · {"KEPT" if r["kept"] else "discarded"}', quote=True)
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="pt {cls}" data-tip="{tip}"/>')
        g.append(f'<rect x="{x - 10:.1f}" y="{PAD["t"]}" width="20" height="{H - PAD["t"] - PAD["b"]}" class="hit" data-tip="{tip}"/>')
    g.append(f'<line x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{H - PAD["b"]}" y2="{H - PAD["b"]}" class="axis"/>')
    g.append(f'<text x="{(PAD["l"] + W - PAD["r"]) / 2:.0f}" y="{H - 12}" class="tick" text-anchor="middle">experiment, in run order ({n})</text>')
    g.append(f'<text x="{PAD["l"]}" y="16" class="title">{html.escape(title)}</text>')
    legend = ('<div class="legend"><span><i class="sw kept"></i>kept</span><span><i class="sw disc"></i>discarded</span>'
              '<span><i class="sw line"></i>running best (kept)</span></div>') if show_legend else ""
    return f'<figure><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{"".join(g)}</svg>{legend}</figure>'


# Reference scores on JevBench public. "exact" = that system's own public-231 run; "est" = leaderboard tier
# accuracies (public + sealed items) re-weighted to the public tier mix 48/72/111, so ±0.02 is plausible.
REFERENCES = [
    {"label": "Strands Decider 2B v19 (AWS)", "acc": 0.723, "kind": "exact", "note": "167/231, strands-decider evaluation/jevbench.md"},
    {"label": "Kev 0.8B (Jared Palmer)", "acc": 0.636, "kind": "exact", "note": "147/231, kev runs/jevbench-public"},
    {"label": "GPT-6 Luna, medium reasoning (LLM baseline)", "acc": 0.993, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Gemini 3.1 Flash-Lite (LLM baseline)", "acc": 0.877, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Mapika decider-4b v2", "acc": 0.833, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Kev 8B (research preview)", "acc": 0.724, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Mapika decider-2b", "acc": 0.701, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Kev 4B (research preview)", "acc": 0.697, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
    {"label": "Laya (ModernBERT-large 421M)", "acc": 0.587, "kind": "est", "note": "JevBench v1.4.2 leaderboard tiers"},
]
TARGET = 0.723


def jev_panel(j, refs=REFERENCES, target=TARGET):
    if not j:
        return ""
    ours = [{"label": LABELS.get(r["run"], r["run"]), "acc": r["acc"], "ours": True,
             "tip": f'{LABELS.get(r["run"], r["run"])}: {r["n"]}/231 = {r["acc"]:.3f} · hard {r["hard"]} · Brier {r["brier"]:.3f}', "val": f'{r["n"]}/231 · {r["acc"]:.3f}'} for r in j]
    theirs = [{"label": r["label"], "acc": r["acc"], "ours": False, "tip": f'{r["label"]}: {r["acc"]:.3f} ({r["kind"]}) · {r["note"]}',
               "val": f'{r["acc"]:.3f}' + ("" if r["kind"] == "exact" else " est.")} for r in refs]
    rows = sorted(ours + theirs, key=lambda r: -r["acc"])
    rowh, top, left, right = 24, 34, 300, 24
    h = top + rowh * len(rows) + 46
    lo, hi = 0.30, 1.00
    def x(v):
        return left + (W - left - right) * (v - lo) / (hi - lo)
    g = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        g.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{h - 40}" class="grid"/>')
        g.append(f'<text x="{x(t):.1f}" y="{h - 22}" class="tick" text-anchor="middle">{t:.1f}</text>')
    g.append(f'<line x1="{x(target):.1f}" x2="{x(target):.1f}" y1="{top - 14}" y2="{h - 40}" class="target"/>')
    g.append(f'<text x="{x(target) + 5:.1f}" y="{top - 16}" class="lbl">goal: beat {target:.3f}</text>')
    for i, r in enumerate(rows):
        y = top + rowh * i
        cls = "bar ours" if r["ours"] else "bar ref"
        g.append(f'<text x="{left - 10}" y="{y + 18}" class="rowlbl{"" if r["ours"] else " muted"}" text-anchor="end">{html.escape(r["label"])}</text>')
        g.append(f'<rect x="{x(lo):.1f}" y="{y + 5}" width="{max(x(r["acc"]) - x(lo), 1):.1f}" height="18" rx="3" class="{cls}" data-tip="{html.escape(r["tip"], quote=True)}"/>')
        g.append(f'<text x="{x(r["acc"]) + 6:.1f}" y="{y + 18}" class="val">{html.escape(r["val"])}</text>')
    g.append(f'<text x="{left - 10}" y="18" class="title" text-anchor="end">accuracy on the 231 public decisions</text>')
    legend = '<div class="legend"><span><i class="sw ours"></i>this repo</span><span><i class="sw ref"></i>published / leaderboard (est. = tier accuracies re-weighted to the public mix)</span><span><i class="sw dash"></i>goal</span></div>'
    return f'<figure><svg viewBox="0 0 {W} {h}" role="img" aria-label="JevBench public accuracy, ranked, with reference systems">{"".join(g)}</svg>{legend}</figure>'


# ---------------------------------------------------------------- page
rows = load_rows()
jev = load_jevbench()
A = {t: [r for r in rows if r["tier"] == "A" and r["torso"] == t] for t in ("Qwen3.5-0.8B", "Qwen3.5-2B", "LFM2.5-230M", "ModernBERT-large")}
B = [r for r in rows if r["tier"] == "B"]
kept_total = sum(r["kept"] for r in rows)
best_a = max((r["sel"] for r in A["Qwen3.5-0.8B"] if r["kept"]), default=0)
best_b = max((r["sel"] for r in B if r["torso"] == "Qwen3.5-0.8B"), default=0)
best_j = max(jev, key=lambda r: r["acc"]) if jev else None

table_rows = "".join(
    f'<tr><td class="mono">{html.escape(r["ts"][5:16])}</td><td>{r["tier"]}</td><td>{html.escape(r["torso"])}</td><td>{html.escape(r["note"][:110])}</td>'
    f'<td class="num">{r["sel"]:.4f}</td><td class="num">{r["acc"]:.3f}</td><td class="num">{r["brier"]:.3f}</td><td class="num">{r["held"]:.3f}</td><td>{"kept" if r["kept"] else "—"}</td></tr>'
    for r in rows)

page = f'''<title>Decider Research Tracker</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  /* Layout: a stat strip, then one chart per torso/tier stacked in a single column (small multiples share the y scale), the JevBench bars, then the full table. */
  :root {{ --bg:#f9f9f7; --surface:#fcfcfb; --fg:#0b0b0b; --fg2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
          --kept:#2a78d6; --disc:#eb6834; --ref:#898781; --body:"IBM Plex Sans",Arial,sans-serif; --mono:"IBM Plex Mono",Menlo,monospace; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --kept:#3987e5; --disc:#d95926; color-scheme:dark; }} }}
  :root[data-theme="dark"] {{ --bg:#0d0d0d; --surface:#1a1a19; --fg:#ffffff; --fg2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --kept:#3987e5; --disc:#d95926; color-scheme:dark; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; margin:0; }}
  .wrap {{ max-width:900px; margin:0 auto; padding-inline:20px; padding-block:32px 64px; }}
  h1 {{ font-size:26px; font-weight:600; margin:0 0 6px; }} h2 {{ font-size:18px; font-weight:600; margin:36px 0 10px; }}
  p {{ max-width:72ch; color:var(--fg2); margin:0 0 12px; }} .muted {{ color:var(--muted); }}
  .strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:16px 0 8px; }}
  .tile {{ background:var(--surface); border:1px solid var(--grid); border-radius:5px; padding:10px 12px; min-width:0; }}
  .tile .k {{ font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
  .tile .v {{ font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }} .tile .s {{ font-size:13px; color:var(--fg2); }}
  figure {{ margin:0 0 8px; background:var(--surface); border:1px solid var(--grid); border-radius:5px; padding:8px 8px 4px; }}
  svg {{ width:100%; height:auto; display:block; font-family:var(--body); }}
  .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }} .ref {{ stroke:var(--ref); stroke-width:1; stroke-dasharray:3 3; }}
  .tick {{ font-size:11px; fill:var(--muted); }} .title {{ font-size:13px; font-weight:600; fill:var(--fg); }} .lbl {{ font-size:11px; fill:var(--fg2); }} .val {{ font-size:11px; fill:var(--fg); font-variant-numeric:tabular-nums; }}
  .best {{ fill:none; stroke:var(--fg); stroke-width:2; stroke-linejoin:round; }}
  .pt.kept {{ fill:var(--kept); stroke:var(--surface); stroke-width:2; }} .pt.disc {{ fill:var(--surface); stroke:var(--disc); stroke-width:2; }}
  .bar.ours {{ fill:var(--kept); }} .bar.ref {{ fill:var(--muted); opacity:.55; }} .target {{ stroke:var(--fg); stroke-width:1.5; stroke-dasharray:5 4; }}
  .rowlbl {{ font-size:12px; fill:var(--fg); }} .rowlbl.muted {{ fill:var(--fg2); }} .hit {{ fill:transparent; }} .hit:hover + * , .pt:hover {{ filter:brightness(1.15); }}
  .sw.ours {{ background:var(--kept); border-radius:2px; }} .sw.ref {{ background:var(--muted); opacity:.55; border-radius:2px; }} .sw.dash {{ border-radius:0; height:0; width:14px; border-top:2px dashed var(--fg); vertical-align:3px; background:none; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:var(--fg2); padding:6px 8px 4px; }}
  .sw {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
  .sw.kept {{ background:var(--kept); }} .sw.disc {{ background:var(--surface); border:2px solid var(--disc); }} .sw.line {{ border-radius:0; height:2px; width:14px; background:var(--fg); vertical-align:3px; }}
  #tip {{ position:fixed; pointer-events:none; background:var(--fg); color:var(--bg); font-size:12px; padding:6px 8px; border-radius:4px; max-width:360px; white-space:pre-wrap; display:none; z-index:9; }}
  .table-wrap {{ overflow-x:auto; border:1px solid var(--grid); border-radius:5px; background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; min-width:760px; }} th,td {{ text-align:left; padding:6px 9px; border-bottom:1px solid var(--grid); vertical-align:top; }}
  th {{ font-family:var(--mono); font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); background:var(--bg); }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }} td.mono {{ font-family:var(--mono); white-space:nowrap; }}
  tr:last-child td {{ border-bottom:none; }}
</style>
<div class="wrap">
  <h1>Decider Research Tracker</h1>
  <p>Every experiment of the decision-model search, in run order. Goal 2 (since 3 Oct evening): beat Strands Decider 2B on JevBench public, 167 of 231. The y axis is the dev selection score (mean over question types of ½[(acc − chance)/(1 − chance) + (1 − Brier)]; higher is better, 1.0 is perfect). Filled points passed the keep gate (more than two noise floors above the pinned baseline); hollow points were discarded. The black line is the best kept score so far. Hover a point for its note.</p>
  <div class="strip">
    <div class="tile"><div class="k">Experiments</div><div class="v">{len(rows)}</div><div class="s">{kept_total} kept</div></div>
    <div class="tile"><div class="k">Tier A best, frozen 0.8B</div><div class="v">{best_a:.3f}</div><div class="s">from 0.287 at the start</div></div>
    <div class="tile"><div class="k">Tier B best, 0.8B + LoRA</div><div class="v">{best_b:.3f}</div><div class="s">dev selection</div></div>
    <div class="tile"><div class="k">JevBench best</div><div class="v">{best_j["n"] if best_j else "—"} / 231</div><div class="s">{(LABELS.get(best_j["run"], best_j["run"]) if best_j else "")}</div></div>
  </div>
  <h2>Tier A: head search on cached frozen-torso states</h2>
  {panel(A["Qwen3.5-0.8B"], "Qwen3.5-0.8B-Base, frozen torso", 0.15, 0.50, True)}
  {panel(A["Qwen3.5-2B"], "Qwen3.5-2B-Base, frozen torso (depth sweep: taps 24, 16, 12, 20 in run order)", 0.15, 0.50, False)}
  {panel(A["LFM2.5-230M"], "LFM2.5-230M-Base, frozen torso", 0.15, 0.50, False)}
  {panel(A["ModernBERT-large"], "ModernBERT-large, frozen torso", 0.15, 0.50, False)}
  <h2>Tier B: LoRA on the torso + head</h2>
  {panel(B, "All torsos, LoRA rank 16, 8k rows", 0.40, 0.70, True)}
  <h2>JevBench public (231 decisions)</h2>
  <p>Ranked list of every model this repo has scored (blue) next to published reference systems (grey). The dashed line is the goal: Strands Decider 2B at 0.723. Reference bars marked est. come from leaderboard tier accuracies that include sealed items, re-weighted to the public tier mix, so read them as ±0.02.</p>
  {jev_panel(jev)}
  <h2>All experiments</h2>
  <div class="table-wrap"><table><thead><tr><th>when</th><th>tier</th><th>torso</th><th>note</th><th class="num">selection</th><th class="num">acc</th><th class="num">Brier</th><th class="num">held-out</th><th>gate</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</div>
<div id="tip"></div>
<script>
  const tip = document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(el => {{
    el.addEventListener('mousemove', e => {{ tip.textContent = el.dataset.tip; tip.style.display = 'block'; tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 380) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }});
    el.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  }});
</script>
'''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page)
print(f"wrote {OUT} with {len(rows)} experiments ({kept_total} kept), {len(jev)} JevBench runs")
