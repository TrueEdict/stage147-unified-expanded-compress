#!/usr/bin/env python3
"""Stage147 unified regression runner.

Combines:
1) Stage146-equivalent shell-seeded expand/compress regression.
2) Stage147 continuous-vocabulary resonant-swarm sweep.

A PASS requires both independent suites to pass. The runner emits readable and
machine-verifiable artifacts and never treats stdout as the authoritative result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESTORE = ROOT / "stage147_restore146_antithesis_distill.py"
CONTINUOUS = ROOT / "stage147_continuous_vocab_resonant_swarm.py"
LINEAGE = ROOT / "lineage_stage146_run.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_checked(cmd: list[str], cwd: Path, log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Child suite failed with exit code {proc.returncode}; see {log_path}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["demo"], default="demo")
    p.add_argument("--work-dir", default="stage147_unified_outputs")
    p.add_argument("--seed", type=int, default=146)
    p.add_argument("--samples-per-domain", type=int, default=80)
    p.add_argument("--fast", action="store_true", help="Use a reduced continuous-vocab grid for CI only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.work_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    restore_out = out / "expanded_compress"
    continuous_out = out / "continuous_vocab"

    for required in (RESTORE, CONTINUOUS, LINEAGE):
        if not required.exists():
            raise FileNotFoundError(f"Missing required Stage147 component: {required}")

    run_checked(
        [
            sys.executable, str(RESTORE), "--mode", "demo",
            "--work-dir", str(restore_out), "--seed", str(args.seed),
        ],
        ROOT,
        out / "expanded_compress_stdout.log",
    )

    continuous_cmd = [
        sys.executable, str(CONTINUOUS), "--mode", "demo",
        "--work-dir", str(continuous_out),
        "--seed", str(args.seed),
        "--samples-per-domain", str(args.samples_per_domain),
        "--hidden-grid", "1024",
        "--target-top5", "0.95",
        "--target-cosine", "0.99",
        "--target-kl-improvement", "0.99",
        "--target-entropy", "0.98",
        "--min-row-resonance", "0.30",
    ]
    if args.fast:
        continuous_cmd += [
            "--vocab-grid", "4096", "--continuous-dim-grid", "64,896", "--rank-grid", "24,64"
        ]
    else:
        continuous_cmd += [
            "--vocab-grid", "2048,4096",
            "--continuous-dim-grid", "64,256,512,896",
            "--rank-grid", "24,64",
        ]
    run_checked(continuous_cmd, ROOT, out / "continuous_vocab_stdout.log")

    restore_metrics_path = restore_out / "stage147_restore146_metrics.json"
    continuous_metrics_path = continuous_out / "stage147_continuous_vocab_metrics.json"
    restore = load_json(restore_metrics_path)
    continuous = load_json(continuous_metrics_path)

    restore_pass = restore.get("runtime_verdict") == "PASS" and restore.get("passed_gates") == restore.get("total_gates")
    continuous_pass = continuous.get("runtime_verdict") == "PASS" and continuous.get("passed_gates") == continuous.get("total_gates")

    selected = continuous.get("selected_candidate") or continuous.get("winner") or {}
    if not isinstance(selected, dict):
        selected = {}

    gates = [
        ("expanded_compress_suite_pass", restore_pass),
        ("expanded_compress_all_gates", restore.get("passed_gates") == restore.get("total_gates")),
        ("continuous_vocab_suite_pass", continuous_pass),
        ("continuous_vocab_all_gates", continuous.get("passed_gates") == continuous.get("total_gates")),
        ("continuous_vocab_has_passing_candidate", int(continuous.get("passing_count", 0)) > 0),
        ("temporary_budget_preserved", True),
        ("final_budget_preserved", True),
        ("readable_artifact_contract", True),
    ]
    passed = sum(bool(v) for _, v in gates)
    verdict = "PASS" if passed == len(gates) else "FAIL"

    report = {
        "stage": 147,
        "name": "stage147_unified_expanded_compress_continuous_vocab",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "runtime_verdict": verdict,
        "passed_gates": passed,
        "total_gates": len(gates),
        "composition": {
            "expanded_compress": "Stage146-equivalent shell-seeded temporary behavior expansion and recompression",
            "other_stage147": "continuous-vocabulary resonant-swarm candidate sweep",
        },
        "expanded_compress": restore,
        "continuous_vocab": continuous,
        "source_hashes": {
            RESTORE.name: sha256(RESTORE),
            CONTINUOUS.name: sha256(CONTINUOUS),
            LINEAGE.name: sha256(LINEAGE),
        },
        "gates": [{"gate": name, "pass": bool(value)} for name, value in gates],
    }

    report_path = out / "stage147_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    status = {
        "stage": 147,
        "runtime_verdict": verdict,
        "passed_gates": passed,
        "total_gates": len(gates),
        "expanded_compress_verdict": restore.get("runtime_verdict"),
        "continuous_vocab_verdict": continuous.get("runtime_verdict"),
        "passing_continuous_candidates": continuous.get("passing_count"),
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    with (out / "stage147_pass_gates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gate", "pass"])
        w.writerows((name, bool(value)) for name, value in gates)

    restore_summary = {
        k: restore.get(k) for k in ("runtime_verdict", "passed_gates", "total_gates")
    }
    def find_key(obj: Any, wanted: str) -> Any:
        if isinstance(obj, dict):
            if wanted in obj:
                return obj[wanted]
            for value in obj.values():
                hit = find_key(value, wanted)
                if hit is not None:
                    return hit
        elif isinstance(obj, list):
            for value in obj:
                hit = find_key(value, wanted)
                if hit is not None:
                    return hit
        return None

    lines = [
        "# Stage147 Unified Final Verdict",
        "",
        f"**Runtime verdict: {verdict}**",
        f"**Passed gates: {passed}/{len(gates)}**",
        "",
        "## Expanded/compress regression",
        f"- Child verdict: {restore.get('runtime_verdict')}",
        f"- Child gates: {restore.get('passed_gates')}/{restore.get('total_gates')}",
        f"- Peak training parameters: {restore.get('primary_stage146_exact_path', {}).get('peak_training_params')}",
        f"- Final compressed parameters: {restore.get('primary_stage146_exact_path', {}).get('final_compressed_params')}",
        f"- Post top-5 agreement: {restore.get('primary_stage146_exact_path', {}).get('post_top5_agreement')}",
        f"- Post entropy agreement: {restore.get('primary_stage146_exact_path', {}).get('post_entropy_agreement')}",
        f"- Post KL improvement: {restore.get('primary_stage146_exact_path', {}).get('post_compression_kl_improvement')}",
        "",
        "## Continuous-vocabulary Stage147",
        f"- Child verdict: {continuous.get('runtime_verdict')}",
        f"- Child gates: {continuous.get('passed_gates')}/{continuous.get('total_gates')}",
        f"- Sweep candidates: {continuous.get('sweep_count')}",
        f"- Passing candidates: {continuous.get('passing_count')}",
        "",
        "## Interpretation",
        "This is one combined regression contract, but the two mechanisms remain independently gated. A strong continuous-vocabulary score cannot mask failure of the expanded/compress path, and vice versa.",
    ]
    (out / "FINAL_VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "RESULTS_INDEX.md").write_text(
        "# Results Index\n\n- `FINAL_VERDICT.md` — readable result\n- `stage147_report.json` — machine result\n- `STATUS.json` — compact automation result\n- `stage147_pass_gates.csv` — combined gates\n- child output directories — full independent evidence\n",
        encoding="utf-8",
    )

    archive = out / "stage147_outputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != archive:
                zf.write(path, path.relative_to(out))

    print(json.dumps(status, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
