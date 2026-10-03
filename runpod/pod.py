"""Manage the RunPod RTX 3090 pod from the laptop: create, status, ssh, stop, terminate.

Uses RunPod's GraphQL API (https://docs.runpod.io/sdks/graphql/manage-pods) with the key from
RUNPOD_API_KEY (load the repo .env first: `set -a; source .env; set +a`). A dedicated SSH key
~/.ssh/runpod_decider is created on first use and injected via the SSH_PUBLIC_KEY env var, so no
account-wide key is needed. Port 22/tcp is exposed for full SSH (rsync works; the proxied SSH
does not support scp/rsync).

    uv run python runpod/pod.py create          # community-cloud 3090, 20 GB pod volume at /workspace
    uv run python runpod/pod.py status          # desiredStatus, public ip:port
    uv run python runpod/pod.py ssh [cmd...]    # ssh in, or run one command
    uv run python runpod/pod.py sync            # rsync repo (+ data/cache) to /workspace/decider
    uv run python runpod/pod.py stop            # stop billing GPU; volume persists
    uv run python runpod/pod.py terminate       # delete pod and its volume

Pod id is remembered in runs/runpod_pod.json. Cost note: a stopped pod bills only its volume
(~$0.10/GB/month); a running idle pod bills the full $0.22/hr.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "runpod_pod.json"
KEY = Path.home() / ".ssh" / "runpod_decider"
TEMPLATE = json.loads((ROOT / "runpod" / "template.json").read_text())


def gql(query: str) -> dict:
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY not set (set -a; source .env; set +a)")
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}", data=json.dumps({"query": query}).encode(),
                                 headers={"content-type": "application/json", "user-agent": "curl/8.0 decider-autoresearch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API HTTP {e.code}: {e.read().decode()[:500]}")
    if out.get("errors"):
        sys.exit(f"RunPod API error: {out['errors']}")
    return out["data"]


def ensure_key() -> str:
    if not KEY.exists():
        subprocess.check_call(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "runpod-decider", "-f", str(KEY)])
    return KEY.with_suffix(".pub").read_text().strip()


def pod_id() -> str:
    if not STATE.exists():
        sys.exit("no pod recorded; run `create` first")
    return json.loads(STATE.read_text())["id"]


def create():
    pub = ensure_key()
    env = {k: v for k, v in TEMPLATE["env"].items() if not v.startswith("<")}
    env["SSH_PUBLIC_KEY"] = pub
    env["PUBLIC_KEY"] = pub          # runpod/* images start sshd from this one
    env_s = ", ".join(f'{{ key: "{k}", value: {json.dumps(v)} }}' for k, v in env.items())
    q = f'''mutation {{ podFindAndDeployOnDemand(input: {{
        cloudType: {TEMPLATE["cloudType"]}, gpuCount: 1, gpuTypeId: {json.dumps(TEMPLATE["gpu"])},
        name: {json.dumps(TEMPLATE["name"])}, imageName: {json.dumps(TEMPLATE["imageName"])},
        volumeInGb: {TEMPLATE["volumeInGb"]}, containerDiskInGb: {TEMPLATE["containerDiskInGb"]},
        minVcpuCount: 4, minMemoryInGb: 16, volumeMountPath: {json.dumps(TEMPLATE["volumeMountPath"])},
        ports: {json.dumps(TEMPLATE["ports"])}, dockerArgs: "", env: [{env_s}] }}) {{ id imageName machineId costPerHr }} }}'''
    d = gql(q)["podFindAndDeployOnDemand"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=1))
    print(f"created pod {d['id']} on {d.get('machineId')} at ${d.get('costPerHr')}/hr")


def status(quiet=False) -> dict:
    d = gql(f'''query {{ pod(input: {{ podId: "{pod_id()}" }}) {{ id name desiredStatus costPerHr
        runtime {{ uptimeInSeconds ports {{ ip isIpPublic privatePort publicPort type }} }} }} }}''')["pod"]
    if not quiet:
        print(json.dumps(d, indent=1))
    return d


def ssh_target(wait_s: int = 600) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < wait_s:
        d = status(quiet=True)
        for p in ((d.get("runtime") or {}).get("ports") or []):
            if p["privatePort"] == 22 and p["type"] == "tcp" and p["isIpPublic"]:
                return p["ip"], int(p["publicPort"])
        time.sleep(10)
    sys.exit("pod has no public tcp:22 yet")


def ssh(args: list[str]):
    ip, port = ssh_target()
    cmd = ["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-p", str(port), f"root@{ip}", *args]
    os.execvp("ssh", cmd)


def sync(with_cache: bool = True):
    ip, port = ssh_target()
    rsh = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
    excludes = ["--exclude", ".venv", "--exclude", "__pycache__", "--exclude", "runs/", "--exclude", ".superpowers", "--exclude", ".env",
                "--exclude", "data/raw", "--exclude", "jevbench/results", "--exclude", "jevbench/.git", "--exclude", ".claude",
                "--exclude", "results*.tsv"]   # the pod appends its own rows; pull them, never push over them
    if not with_cache:
        excludes += ["--exclude", "data/cache"]
    subprocess.check_call(["rsync", "-az", "--stats", "-e", rsh, *excludes, f"{ROOT}/", f"root@{ip}:/workspace/decider/"])
    print("synced")


def stop():
    print(gql(f'mutation {{ podStop(input: {{ podId: "{pod_id()}" }}) {{ id desiredStatus }} }}'))


def terminate():
    gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id()}" }}) }}')
    STATE.unlink(missing_ok=True)
    print("terminated")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["status"]
    {"create": create, "status": status, "ssh": lambda: ssh(rest), "sync": lambda: sync("--no-cache" not in rest),
     "stop": stop, "terminate": terminate}[cmd]()
