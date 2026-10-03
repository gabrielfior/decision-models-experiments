"""Print this month's Modal COMPUTE spend (metered cost minus storage), used as the $5 gate."""
import argparse
import json
import subprocess
import sys


def compute_spend_usd() -> float:
    out = subprocess.check_output(["modal", "billing", "summary", "--json"]).decode()
    j = json.loads(out)
    metered = float(j["metered_cost"])
    storage = float(j.get("metered_cost_breakdown", {}).get("volumes", 0))
    return metered - storage


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", action="store_true", help="print the bare number only")
    a = ap.parse_args()
    v = compute_spend_usd()
    print(f"{v:.4f}" if a.value else f"Modal compute spend this month: ${v:.2f}")
    sys.exit(0)
