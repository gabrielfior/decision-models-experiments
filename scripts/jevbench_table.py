"""Regenerate docs/jevbench-public.tsv from every runs/jevbench/<name>/summary.json (+ results.jsonl for split counts).
usage: uv run python scripts/jevbench_table.py [--out docs/jevbench-public.tsv]"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["run", "accuracy", "n_correct", "brier", "ece", "ordinal_mae", "paraphrase_agreement", "easy", "standard", "hard", "p50_s"]


def row_for(d: Path) -> dict | None:
    s_path, r_path = d / "summary.json", d / "results.jsonl"
    if not s_path.exists():
        return None
    s = json.loads(s_path.read_text())
    counts = {"easy": [0, 0], "original": [0, 0], "hard": [0, 0]}
    if r_path.exists():
        for line in r_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = r["task_id"].split("-")[0]
            if k in counts:
                counts[k][1] += 1
                counts[k][0] += int(bool(r.get("correct")))
    ece = s.get("ece")
    ece = ece.get("ece") if isinstance(ece, dict) else ece
    par = s.get("paraphrase_consistency")
    par = par.get("agreement") if isinstance(par, dict) else par
    return {"run": d.name, "accuracy": f"{s['accuracy']:.4f}", "n_correct": s["n_correct"], "brier": f"{s['brier_mean']:.4f}",
            "ece": f"{ece:.4f}" if ece is not None else "", "ordinal_mae": f"{s['ordinal_mae']:.4f}" if s.get("ordinal_mae") is not None else "",
            "paraphrase_agreement": f"{par:.3f}" if par is not None else "",
            "easy": f"{counts['easy'][0]}/{counts['easy'][1]}", "standard": f"{counts['original'][0]}/{counts['original'][1]}",
            "hard": f"{counts['hard'][0]}/{counts['hard'][1]}", "p50_s": f"{(s.get('latency') or {}).get('p50_s', float('nan')):.2f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "docs" / "jevbench-public.tsv", type=Path)
    a = ap.parse_args()
    rows = [r for d in sorted((ROOT / "runs" / "jevbench").iterdir()) if d.is_dir() and not d.name.startswith("smoke") and (r := row_for(d))]
    a.out.write_text("\t".join(COLS) + "\n" + "\n".join("\t".join(str(r[c]) for c in COLS) for r in rows) + "\n")
    print(f"{len(rows)} scorings -> {a.out}")


if __name__ == "__main__":
    main()
