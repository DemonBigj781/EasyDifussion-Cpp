#!/usr/bin/env python3
"""Source-only audit for the shared GGML merge contract.

This complements ``audit_symbols.py``. The symbol audit needs a completed build;
this script inspects the checkout before building so task 32 can catch stale
PIN data, unexpected duplicate GGML trees, changed donor files, or missing
compile-layout invariants without running model or GPU workloads.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "cosmopolitan" / "shared-ggml"
PIN = SHARED / "PIN.json"
PREPARE = SHARED / "prepare.py"

REQUIRED_SHARED_DEFINITIONS = {
    "all_consumers": "GGML_MAX_NAME=160",
    "ggml": "GGML_COSMO_STATIC_ONLY",
}

REQUIRED_PREPARE_MARKERS = (
    "GGML_MAX_NAME=160",
    "GGML_COSMO_STATIC_ONLY",
    "llama_donor_files",
    "authoritative_tree_sha256",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(value: object) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE)


def tracked_entries(prefix: str) -> list[dict[str, str]]:
    result = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z", "--", prefix.rstrip("/") + "/"],
        cwd=ROOT,
    )
    entries: list[dict[str, str]] = []
    full_prefix = prefix.rstrip("/") + "/"
    for item in result.split(b"\0"):
        if not item:
            continue
        meta, full_path = item.split(b"\t", 1)
        mode, _blob, stage = meta.decode().split()
        if stage != "0":
            raise RuntimeError(f"Unmerged tracked entry under {prefix}: {full_path!r}")
        path = full_path.decode()
        relative = path[len(full_prefix):]
        entry = {"path": relative, "mode": mode}
        entry["sha256"] = content_hash(ROOT / full_prefix / relative, mode)
        entries.append(entry)
    return entries


def content_hash(path: Path, mode: str) -> str:
    if mode == "120000":
        if not path.is_symlink():
            raise RuntimeError(f"Expected symlink tracked entry: {path}")
        return digest(path.readlink().as_posix().encode())
    if mode not in ("100644", "100755"):
        raise RuntimeError(f"Unsupported tracked entry mode {mode}: {path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Expected regular tracked file: {path}")
    return digest(path.read_bytes())


def tree_summary(prefix: str) -> dict[str, object]:
    entries = tracked_entries(prefix)
    return {
        "path": prefix,
        "tracked_file_count": len(entries),
        "tree_sha256": stable_hash(entries),
    }


def tracked_ggml_roots() -> list[str]:
    files = git_output(["ls-files"]).splitlines()
    roots = set()
    for path in files:
        marker = "/ggml/"
        if marker not in path:
            continue
        root = path.split(marker, 1)[0] + "/ggml"
        if path.startswith("cosmopolitan/shared-ggml/"):
            continue
        roots.add(root)
    return sorted(roots)


def digest_file(relative: str) -> str:
    return digest((ROOT / relative).read_bytes())


def compare_files(authoritative: str, llama: str, relative_paths: Iterable[str]) -> list[dict[str, object]]:
    comparisons = []
    for relative in sorted(set(relative_paths)):
        left = ROOT / authoritative / relative
        right = ROOT / llama / relative
        if not left.exists() or not right.exists() or not left.is_file() or not right.is_file():
            comparisons.append({"path": relative, "status": "missing"})
            continue
        left_hash = digest(left.read_bytes())
        right_hash = digest(right.read_bytes())
        comparisons.append({
            "path": relative,
            "status": "same" if left_hash == right_hash else "different",
            "authoritative_sha256": left_hash,
            "llama_sha256": right_hash,
        })
    return comparisons


def collect_audit() -> dict[str, object]:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    authoritative = pin["authoritative_source"]
    llama = pin["llama_source"]
    authoritative_summary = tree_summary(authoritative)
    llama_summary = tree_summary(llama)
    donor_results = {}
    for relative, expected in sorted(pin["llama_donor_files"].items()):
        actual = digest_file(f"{llama}/{relative}")
        donor_results[relative] = {"expected_sha256": expected, "actual_sha256": actual, "matches_pin": actual == expected}
    patch_results = {}
    for name, expected in sorted(pin["patches"].items()):
        actual = digest_file(f"cosmopolitan/shared-ggml/patches/{name}")
        patch_results[name] = {"expected_sha256": expected, "actual_sha256": actual, "matches_pin": actual == expected}
    compare_targets = ["include/ggml.h", "include/ggml-backend.h", "src/ggml.c", "src/ggml.cpp"]
    compare_targets.extend(pin["llama_donor_files"].keys())
    prepare_text = PREPARE.read_text(encoding="utf-8")
    roots = tracked_ggml_roots()
    expected_roots = sorted({authoritative, llama})
    return {
        "schema": 1,
        "scope": "source-only; no build, application launch, model load, or GPU execution",
        "git": {
            "head": git_output(["rev-parse", "--short", "HEAD"]).strip(),
            "branch": git_output(["branch", "--show-current"]).strip(),
        },
        "pin": {
            "path": PIN.relative_to(ROOT).as_posix(),
            "schema": pin.get("schema"),
            "authoritative_source": authoritative,
            "authoritative_ggml_base_revision": pin["authoritative_ggml_base_revision"],
            "llama_source": llama,
            "llama_revision": pin["llama_revision"],
            "ggml_max_name": pin["ggml_max_name"],
        },
        "trees": {
            "authoritative": {
                **authoritative_summary,
                "expected_file_count": pin["authoritative_file_count"],
                "expected_tree_sha256": pin["authoritative_tree_sha256"],
                "matches_pin": (
                    authoritative_summary["tracked_file_count"] == pin["authoritative_file_count"]
                    and authoritative_summary["tree_sha256"] == pin["authoritative_tree_sha256"]
                ),
            },
            "llama": llama_summary,
        },
        "donor_files": donor_results,
        "patches": patch_results,
        "ggml_roots": {
            "tracked": roots,
            "expected": expected_roots,
            "unexpected": [root for root in roots if root not in expected_roots],
            "expected_missing": [root for root in expected_roots if root not in roots],
        },
        "authoritative_vs_llama": compare_files(authoritative, llama, compare_targets),
        "prepare_contract": {
            "path": PREPARE.relative_to(ROOT).as_posix(),
            "required_markers": {marker: marker in prepare_text for marker in REQUIRED_PREPARE_MARKERS},
            "required_definitions": REQUIRED_SHARED_DEFINITIONS,
        },
        "gpu_execution": {"status": "deferred", "reason": "task 32 source audit only"},
    }


def validate(audit: dict[str, object]) -> list[str]:
    errors: list[str] = []
    trees = audit["trees"]
    if not trees["authoritative"]["matches_pin"]:
        errors.append("authoritative GGML tree does not match PIN.json")
    for path, result in audit["donor_files"].items():
        if not result["matches_pin"]:
            errors.append(f"llama donor changed: {path}")
    for name, result in audit["patches"].items():
        if not result["matches_pin"]:
            errors.append(f"shared GGML patch hash mismatch: {name}")
    roots = audit["ggml_roots"]
    if roots["unexpected"]:
        errors.append("unexpected tracked GGML roots: " + ", ".join(roots["unexpected"]))
    if roots["expected_missing"]:
        errors.append("expected GGML roots missing: " + ", ".join(roots["expected_missing"]))
    markers = audit["prepare_contract"]["required_markers"]
    missing = [marker for marker, present in markers.items() if not present]
    if missing:
        errors.append("prepare.py missing required shared-GGML markers: " + ", ".join(missing))
    if audit["pin"]["ggml_max_name"] != 160:
        errors.append("GGML_MAX_NAME pin must remain 160 until both consumers are audited again")
    return errors


def markdown_list(items: Sequence[str]) -> list[str]:
    return [f"- `{item}`" for item in items] if items else ["- none"]


def to_markdown(audit: dict[str, object], errors: Sequence[str]) -> str:
    lines = [
        "# Shared GGML source audit",
        "",
        f"Scope: {audit['scope']}.",
        "",
        "## Result",
        "",
        "PASS" if not errors else "FAIL",
        "",
    ]
    if errors:
        lines.extend(["## Errors", "", *markdown_list(errors), ""])
    pin = audit["pin"]
    lines.extend([
        "## Pin",
        "",
        f"- Authoritative source: `{pin['authoritative_source']}`",
        f"- Authoritative base revision: `{pin['authoritative_ggml_base_revision']}`",
        f"- Llama source: `{pin['llama_source']}`",
        f"- Llama revision: `{pin['llama_revision']}`",
        f"- GGML_MAX_NAME: `{pin['ggml_max_name']}`",
        "",
        "## Trees",
        "",
    ])
    for name in ("authoritative", "llama"):
        tree = audit["trees"][name]
        lines.append(f"- {name}: `{tree['path']}` files={tree['tracked_file_count']} sha256=`{tree['tree_sha256']}`")
    lines.extend([
        "",
        "## Tracked GGML roots",
        "",
        *markdown_list(audit["ggml_roots"]["tracked"]),
        "",
        "## Different authoritative/llama comparison files",
        "",
    ])
    different = [item["path"] for item in audit["authoritative_vs_llama"] if item["status"] != "same"]
    lines.extend(markdown_list(different))
    lines.extend(["", "GPU execution: deferred.", ""])
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-fail", action="store_true", help="write the audit but return success even when validation fails")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    audit = collect_audit()
    errors = validate(audit)
    payload: str
    if args.format == "json":
        payload = json.dumps({"status": "passed" if not errors else "failed", "errors": errors, **audit}, indent=2, sort_keys=True) + "\n"
    else:
        payload = to_markdown(audit, errors)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if args.no_fail or not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
