#!/usr/bin/env python3
"""Inventory route declarations without importing the Python application."""

import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "websocket", "route"}


def literal(node, fallback=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return fallback


def main():
    routes, mounts, errors = [], [], []
    files = sorted((ROOT / "ui/easydiffusion").rglob("*.py"))
    files += sorted((ROOT / "ui/plugins/server").rglob("*.py"))
    files += sorted((ROOT / "training").rglob("*.py"))
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        except (SyntaxError, UnicodeError) as error:
            errors.append({"file": relative, "error": str(error)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                        continue
                    method = decorator.func.attr
                    if method not in METHODS or not decorator.args:
                        continue
                    route = literal(decorator.args[0])
                    if not isinstance(route, str) or not route.startswith("/"):
                        continue
                    methods = [method.upper()]
                    for keyword in decorator.keywords:
                        if keyword.arg == "methods":
                            methods = literal(keyword.value, methods)
                    routes.append({"file": relative, "line": decorator.lineno,
                                   "receiver": ast.unparse(decorator.func.value),
                                   "declared_path": route, "methods": methods,
                                   "handler": node.name})
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "include_router" and node.args):
                prefix = ""
                for keyword in node.keywords:
                    if keyword.arg == "prefix":
                        prefix = literal(keyword.value, "<dynamic>")
                mounts.append({"file": relative, "line": node.lineno,
                               "receiver": ast.unparse(node.func.value),
                               "router": ast.unparse(node.args[0]), "prefix": prefix})
    native = []
    for path in [ROOT / "source/sdkit3-port-source/src/server.cpp",
                 ROOT / "cosmopolitan/src/ui_routes.cpp"]:
        content = path.read_text()
        for match in re.finditer(r'CROW_ROUTE\([^,]+,\s*"([^"\n]+)"\)', content):
            following = content[match.end():match.end() + 160]
            method_match = re.match(r'\.methods\(([^)]*)\)', following)
            methods = re.findall(r'"([A-Z]+)"', method_match.group(1)) if method_match else ["GET"]
            native.append({"file": path.relative_to(ROOT).as_posix(),
                           "line": content.count("\n", 0, match.start()) + 1,
                           "declared_path": match.group(1), "methods": methods})
    report = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Literal route declarations, not a generated OpenAPI schema, active-route count or proof of parity. Parallel legacy/plugin modules may repeat declarations. Router mount prefixes are listed separately; dynamic routes and application middleware require manual review.",
        "python_files_scanned": len(files), "parse_errors": errors,
        "python_route_declarations": routes, "python_router_mounts": mounts,
        "native_literal_route_declarations": native,
        "native_dynamic_routes": "C++ UI page definitions are registered dynamically under /cpp-ui.",
    }
    output = ROOT / "cosmopolitan/docs/api-inventory.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(routes)} Python route declarations; {len(mounts)} router mounts; "
          f"{len(native)} native literal routes; {len(errors)} parse errors")
    if errors:
        raise SystemExit("Incomplete inventory: inspect parse_errors")


if __name__ == "__main__":
    main()
