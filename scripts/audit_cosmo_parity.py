#!/usr/bin/env python3
"""Audit non-runtime parity between active main, Theory, and Cosmo checkouts.

This tool intentionally performs static repository inspection only. It does not
build, start the application, load models, run GPU probes, or contact external
services. Its output is meant to seed the route/file inventory required by
``docs/todo/33-cosmo-main-theory-parity.md``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


DEFAULT_ROOT = Path(
    "/mnt/38FEF88DFEF84522/easy-diffusion-versions/easy-diffusion-cpp"
)
DEFAULT_MAIN = DEFAULT_ROOT / "easy-diffusion-cpp"
DEFAULT_THEORY = DEFAULT_ROOT / "easy-diffusion-cpp-Theory"
DEFAULT_COSMO = DEFAULT_ROOT / "Easy-Difussion-Cpp-Cosmo"

PRODUCTION_PYTHON_GLOBS = (
    "ui/**/*.py",
    "training/**/*.py",
)

KEY_THEORY_PREFIXES = (
    ".dockerfiles/",
    ".github/",
    "Audit/",
    "ci/",
    "docs/",
    "scripts/",
    "source/API.test/",
    "source/DiffUser.cpp/",
    "source/INFERENCE.cpp/",
    "source/RestAPI.cpp/",
    "source/UI.cpp/",
    "ui/",
)

PY_ROUTE_RE = re.compile(
    r"@(?:app|router)\.(?:get|post|put|patch|delete|options|head)\(\s*"
    r"(?P<quote>['\"])(?P<path>[^'\"]+)(?P=quote)"
)
JS_ROUTE_RE = re.compile(
    r"\b(?:app|router)\.(?:get|post|put|patch|delete|options|head)\(\s*"
    r"(?P<quote>['\"])(?P<path>[^'\"]+)(?P=quote)"
)
CROW_ROUTE_RE = re.compile(r"CROW_ROUTE\([^,]+,\s*(?P<quote>['\"])(?P<path>[^'\"]+)(?P=quote)")


def run_git(repo: Path, args: Sequence[str]) -> str:
    command = ["git", "-C", str(repo), *args]
    return subprocess.check_output(command, text=True, stderr=subprocess.PIPE)


def git_lines(repo: Path, args: Sequence[str]) -> list[str]:
    output = run_git(repo, args)
    return [line for line in output.splitlines() if line]


def repo_info(repo: Path) -> dict[str, object]:
    branch = run_git(repo, ["branch", "--show-current"]).strip()
    head = run_git(repo, ["rev-parse", "--short", "HEAD"]).strip()
    files = git_lines(repo, ["ls-files"])
    return {
        "path": str(repo),
        "branch": branch,
        "head": head,
        "tracked_file_count": len(files),
    }


def tracked_files(repo: Path, globs: Sequence[str] | None = None) -> set[str]:
    args = ["ls-files"]
    if globs:
        args.extend(globs)
    return set(git_lines(repo, args))


def topdir_counts(paths: Iterable[str]) -> dict[str, int]:
    counts = Counter(path.split("/", 1)[0] for path in paths)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def python_area_counts(paths: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        parts = path.split("/")
        if not parts:
            continue
        if parts[0] == "ui" and len(parts) > 1:
            counts[f"ui/{parts[1]}"] += 1
        else:
            counts[parts[0]] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def text_for(repo: Path, tracked_path: str) -> str:
    path = repo / tracked_path
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def active_images_v6_refs(repo: Path) -> list[str]:
    refs: list[str] = []
    for path in sorted(tracked_files(repo)):
        if not path.endswith((".py", ".js", ".html", ".md")):
            continue
        text = text_for(repo, path)
        for number, line in enumerate(text.splitlines(), 1):
            if "images_v6" in line or "images-v6" in line:
                refs.append(f"{path}:{number}:{line.strip()}")
    return refs


def model_scoped_civitai_refs(repo: Path) -> list[str]:
    needles = (
        "model-locations",
        "select-model-first",
        "model-scoped",
        "Select a model first",
    )
    refs: list[str] = []
    for path in sorted(tracked_files(repo)):
        if not path.endswith((".py", ".js", ".html", ".md")):
            continue
        text = text_for(repo, path)
        for number, line in enumerate(text.splitlines(), 1):
            if any(needle in line for needle in needles):
                refs.append(f"{path}:{number}:{line.strip()}")
    return refs


def route_declarations(repo: Path) -> dict[str, object]:
    declarations: list[dict[str, str]] = []
    for path in sorted(tracked_files(repo)):
        if not path.endswith((".py", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".js")):
            continue
        text = text_for(repo, path)
        patterns = []
        if path.endswith(".py"):
            patterns.append(("python", PY_ROUTE_RE))
        if path.endswith(".js"):
            patterns.append(("javascript", JS_ROUTE_RE))
        if path.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
            patterns.append(("crow", CROW_ROUTE_RE))
        for kind, pattern in patterns:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                declarations.append(
                    {
                        "kind": kind,
                        "path": match.group("path"),
                        "source": f"{path}:{line}",
                    }
                )
    by_kind = Counter(item["kind"] for item in declarations)
    return {
        "count": len(declarations),
        "by_kind": dict(sorted(by_kind.items())),
        "declarations": declarations,
    }


def collect_audit(main: Path, theory: Path, cosmo: Path) -> dict[str, object]:
    main_files = tracked_files(main)
    theory_files = tracked_files(theory)
    cosmo_files = tracked_files(cosmo)
    main_missing = sorted(main_files - cosmo_files)
    theory_missing = sorted(theory_files - cosmo_files)
    production_python = sorted(tracked_files(cosmo, PRODUCTION_PYTHON_GLOBS))
    key_theory_missing = [
        path for path in theory_missing if path.startswith(KEY_THEORY_PREFIXES)
    ]
    images_v6_refs = active_images_v6_refs(cosmo)
    model_scoped_refs = model_scoped_civitai_refs(cosmo)
    return {
        "repos": {
            "main": repo_info(main),
            "theory": repo_info(theory),
            "cosmo": repo_info(cosmo),
        },
        "file_parity": {
            "main_missing_in_cosmo_count": len(main_missing),
            "main_missing_in_cosmo": main_missing,
            "theory_missing_in_cosmo_count": len(theory_missing),
            "theory_missing_in_cosmo_topdir_counts": topdir_counts(theory_missing),
            "theory_missing_in_cosmo_key_count": len(key_theory_missing),
            "theory_missing_in_cosmo_key": key_theory_missing,
        },
        "python_runtime": {
            "tracked_production_python_count": len(production_python),
            "tracked_production_python_area_counts": python_area_counts(production_python),
            "tracked_production_python": production_python,
        },
        "routes": {
            "main": route_declarations(main),
            "theory": route_declarations(theory),
            "cosmo": route_declarations(cosmo),
        },
        "civitai_model_images": {
            "images_v6_reference_count": len(images_v6_refs),
            "images_v6_references": images_v6_refs,
            "model_scoped_reference_count": len(model_scoped_refs),
            "model_scoped_references": model_scoped_refs,
        },
        "gpu_execution": {
            "status": "deferred",
            "reason": "static audit only; GPU execution tests are intentionally not run",
        },
    }


def markdown_list(items: Sequence[str], *, limit: int = 20) -> str:
    if not items:
        return "- none"
    visible = items[:limit]
    lines = [f"- `{item}`" for item in visible]
    remaining = len(items) - len(visible)
    if remaining > 0:
        lines.append(f"- ... {remaining} more")
    return "\n".join(lines)


def to_markdown(audit: dict[str, object]) -> str:
    repos = audit["repos"]
    parity = audit["file_parity"]
    python_runtime = audit["python_runtime"]
    civitai = audit["civitai_model_images"]
    routes = audit["routes"]
    lines = [
        "# Cosmo parity static audit",
        "",
        "## Repositories",
        "",
        "| Repo | Branch | Head | Tracked files | Path |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name in ("main", "theory", "cosmo"):
        info = repos[name]
        lines.append(
            f"| {name} | `{info['branch']}` | `{info['head']}` | "
            f"{info['tracked_file_count']} | `{info['path']}` |"
        )
    lines.extend(
        [
            "",
            "## File parity",
            "",
            f"- Active-main paths missing in Cosmo: {parity['main_missing_in_cosmo_count']}",
            f"- Theory paths missing in Cosmo: {parity['theory_missing_in_cosmo_count']}",
            f"- Key Theory paths missing in Cosmo: {parity['theory_missing_in_cosmo_key_count']}",
            "",
            "### Active-main paths missing in Cosmo",
            "",
            markdown_list(parity["main_missing_in_cosmo"]),
            "",
            "### Theory missing top-level counts",
            "",
        ]
    )
    for topdir, count in parity["theory_missing_in_cosmo_topdir_counts"].items():
        lines.append(f"- `{topdir}`: {count}")
    lines.extend(
        [
            "",
            "## Python runtime surface",
            "",
            f"Tracked production Python files in Cosmo: "
            f"{python_runtime['tracked_production_python_count']}",
        ]
    )
    for area, count in python_runtime["tracked_production_python_area_counts"].items():
        lines.append(f"- `{area}`: {count}")
    lines.extend(
        [
            "",
            "## Route declaration inventory",
            "",
            "| Repo | Declarations | By kind |",
            "| --- | ---: | --- |",
        ]
    )
    for name in ("main", "theory", "cosmo"):
        route_info = routes[name]
        lines.append(
            f"| {name} | {route_info['count']} | "
            f"`{json.dumps(route_info['by_kind'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Civitai model-scoped image flow",
            "",
            f"- `images_v6` / `images-v6` references: {civitai['images_v6_reference_count']}",
            f"- model-scoped references: {civitai['model_scoped_reference_count']}",
            "",
            "## GPU execution",
            "",
            "Deferred. This audit performs static inspection only.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--theory", type=Path, default=DEFAULT_THEORY)
    parser.add_argument("--cosmo", type=Path, default=DEFAULT_COSMO)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    audit = collect_audit(args.main, args.theory, args.cosmo)
    if args.format == "json":
        output = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    else:
        output = to_markdown(audit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
