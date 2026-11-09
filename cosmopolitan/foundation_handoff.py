#!/usr/bin/env python3
"""Validate a pinned foundation's declared build identity without executing it.

This contract is deliberately not a toolchain selector. The existing build
continues to use its existing pin until a complete source/runtime handoff and
the corresponding compiler, header, CRT and linker integration are ready.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


SCHEMA_VERSION = 1
FORMAT = "easy-diffusion-foundation-handoff"
TOOLS = ("c_compiler", "cxx_compiler", "archiver", "ranlib", "strip",
         "ape_linker", "ape_loader")
COMPONENTS = ("rust_std", "webgpu", "mesa", "llvm")
ARTIFACT_KINDS = ("archive", "object", "executable", "data")


class HandoffError(ValueError):
    """The handoff is incomplete, inconsistent, or differs from its inputs."""


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise HandoffError(f"{label}: expected exactly {', '.join(fields)}")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise HandoffError(f"{label}: expected nonempty, trimmed text")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise HandoffError(f"{label}: control characters are not allowed")
    return value


def _hex(value, length, label):
    if (not isinstance(value, str) or
            re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None or
            len(set(value)) == 1):
        raise HandoffError(f"{label}: expected a real lowercase {length}-digit identity")
    return value


def _revision(value, label):
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise HandoffError(f"{label}: expected a full pinned Git revision")
    return _hex(value, len(value), label)


def _repository(value, label):
    value = _text(value, label)
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", value):
        raise HandoffError(f"{label}: expected an HTTPS repository URL without credentials")
    return value


def _relative(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or
            any(part in (".", "..") for part in path.parts) or
            "\\" in value or ":" in value or not path.parts):
        raise HandoffError(f"{label}: expected a normalized relative POSIX path")
    return value


def _list(value, label, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise HandoffError(f"{label}: expected {'nonempty ' if nonempty else ''}list")
    if not all(isinstance(item, str) for item in value) or len(set(value)) != len(value):
        raise HandoffError(f"{label}: expected distinct string entries")
    return value


def _file(root, relative):
    # Exported artifacts must be ordinary files. Materialize symlinks when
    # exporting a toolchain so the handoff is independent of external paths.
    path = root
    for part in PurePosixPath(relative).parts:
        path /= part
        if path.is_symlink():
            raise HandoffError(f"Artifact symlink is not allowed: {relative}")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise HandoffError(f"Missing regular artifact: {relative}")
    return path


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def _static_archive(path, name):
    """Check a nonempty ordinary ar container without executing toolchain tools.

    This is a container check, not an ELF/bitcode/ABI correctness check. Both
    GNU and BSD member-name encodings are accepted.
    """
    size = path.stat().st_size
    members = 0
    name_table = None
    long_names = []
    with path.open("rb") as source:
        if source.read(8) != b"!<arch>\n":
            raise HandoffError("Expected a complete static archive, not a thin/shared library: " + name)
        offset = 8
        while offset < size:
            header = source.read(60)
            if len(header) != 60 or header[58:60] != b"`\n":
                raise HandoffError("Malformed archive member header: " + name)
            length = header[48:58].strip()
            if not length or not length.isdigit():
                raise HandoffError("Invalid archive member length: " + name)
            length = int(length)
            end = offset + 60 + length
            if end + length % 2 > size:
                raise HandoffError("Truncated archive member: " + name)
            member_name = header[:16].strip()
            payload_length = length
            bsd_name = False
            if member_name.startswith(b"#1/"):
                bsd_name = True
                name_length = member_name[3:]
                if not name_length.isdigit() or int(name_length) > length:
                    raise HandoffError("Invalid BSD archive member name: " + name)
                name_length = int(name_length)
                member_name = source.read(name_length).rstrip(b"\0")
                payload_length -= name_length
            if not bsd_name and member_name == b"//":
                if name_table is not None:
                    raise HandoffError("Duplicate GNU archive name table: " + name)
                name_table = source.read(payload_length)
            elif not bsd_name and member_name.startswith(b"/") and member_name not in (b"/", b"/SYM64/"):
                if not member_name[1:].isdigit():
                    raise HandoffError("Invalid GNU archive member name: " + name)
                long_names.append(int(member_name[1:]))
            if (member_name not in (b"/", b"//", b"/SYM64/") and
                    not member_name.startswith(b"__.SYMDEF")):
                if not member_name or not payload_length:
                    raise HandoffError("Empty archive member: " + name)
                members += 1
            source.seek(end)
            if length % 2 and source.read(1) != b"\n":
                raise HandoffError("Invalid archive member padding: " + name)
            offset = end + length % 2
        if not members:
            raise HandoffError("Static archive has no object/data members: " + name)
    for position in long_names:
        if name_table is None or position >= len(name_table):
            raise HandoffError("Missing/out-of-bounds GNU archive member name: " + name)
        if position and name_table[position - 1:position] != b"\n":
            raise HandoffError("GNU archive member name does not start at an entry: " + name)
        end = name_table.find(b"/\n", position)
        if end <= position or b"\n" in name_table[position:end] or b"\0" in name_table[position:end]:
            raise HandoffError("Invalid GNU archive member name terminator: " + name)


def header_tree_sha256(root, relative):
    """Hash every regular header-tree file, without following symlinks.

    The digest is SHA256 of canonical JSON mapping sorted relative file names
    to their SHA256. Empty trees and non-regular entries are rejected.
    """
    relative = _relative(relative, "runtime.headers.path")
    directory = root
    for part in PurePosixPath(relative).parts:
        directory /= part
        if directory.is_symlink():
            raise HandoffError("Header tree cannot contain a symlink")
    if not directory.is_dir():
        raise HandoffError("Missing runtime header tree")
    entries = {}

    def walk(current):
        # glob/rglob can suppress directory-scanning errors. A missing subtree
        # must fail rather than producing a digest of a partial include tree.
        try:
            with os.scandir(current) as scan:
                children = sorted(scan, key=lambda entry: entry.name)
            for entry in children:
                path = Path(entry.path)
                name = path.relative_to(directory).as_posix()
                _relative(name, "header path")
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise HandoffError(f"Header tree cannot contain a symlink: {path}")
                if stat.S_ISDIR(mode):
                    walk(path)
                elif stat.S_ISREG(mode):
                    entries[name] = _digest(path)
                else:
                    raise HandoffError(f"Non-regular header tree entry: {name}")
        except OSError as error:
            raise HandoffError(f"Cannot read complete header tree at {current}: {error}") from error

    walk(directory)
    if not entries:
        raise HandoffError("Runtime header tree is empty")
    return _canonical_hash(entries)


def runtime_identity_sha256(data):
    """Bind runtime lineage, compiler tools, headers, CRT and C++ ABI bytes.

    Callers producing a manifest can calculate this before filling component
    identity fields. Validation checks all referenced artifact bytes later.
    """
    runtime = data["runtime"]
    toolchain = data["toolchain"]
    names = set(runtime["crt_objects"] + runtime["archives"] +
                runtime["cxx_abi_archives"] + toolchain["support_files"])
    names.add(runtime["provenance_record"])
    names.update(toolchain[name] for name in TOOLS)
    return _canonical_hash({
        "foundation": data["foundation"], "target": data["target"],
        "runtime": runtime, "toolchain": toolchain,
        "artifacts": {name: data["artifacts"][name] for name in sorted(names)},
    })


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise HandoffError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_handoff(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"Cannot read handoff: {error}") from error


def validate_handoff(data, root, *, expected_repository, expected_revision,
                     expected_architecture="x86_64"):
    """Validate source pin, declared target/lineage, and complete exported bytes.

    No compiler, build, model, process or runtime probe is launched. This is
    static input verification, not proof of ABI correctness or model behavior.
    """
    _object(data, ("format", "schema_version", "foundation", "target", "runtime",
                   "toolchain", "runtime_identity_sha256", "components", "artifacts"),
            "handoff")
    if data["format"] != FORMAT or type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise HandoffError("Unsupported foundation handoff format/version")
    expected_repository = _repository(expected_repository, "expected_repository")
    expected_revision = _revision(expected_revision, "expected_revision")
    foundation = _object(data["foundation"],
                         ("repository", "revision", "source_tree_sha256", "clean"),
                         "foundation")
    _repository(foundation["repository"], "foundation.repository")
    _revision(foundation["revision"], "foundation.revision")
    _hex(foundation["source_tree_sha256"], 64, "foundation.source_tree_sha256")
    if foundation["clean"] is not True:
        raise HandoffError("Foundation source must be a clean pinned tree")
    if (foundation["repository"] != expected_repository or
            foundation["revision"] != expected_revision):
        raise HandoffError("Foundation source does not match the selected repository/revision")

    target = _object(data["target"], ("architecture", "abi", "c_triple", "rust_triple"), "target")
    if expected_architecture not in ("x86_64", "aarch64"):
        raise HandoffError("Unsupported requested architecture")
    if target["architecture"] != expected_architecture or target["abi"] != "cosmopolitan-ape":
        raise HandoffError("Foundation target does not match the requested APE architecture")
    for field in ("c_triple", "rust_triple"):
        triple = _text(target[field], "target." + field)
        if not re.fullmatch(re.escape(expected_architecture) + r"-[a-z0-9_-]+", triple):
            raise HandoffError(f"target.{field}: triple must match the requested architecture")

    artifacts = data["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise HandoffError("artifacts: expected a nonempty artifact map")
    for name, entry in artifacts.items():
        _relative(name, "artifact path")
        _object(entry, ("sha256", "kind"), "artifact " + name)
        _hex(entry["sha256"], 64, "artifact digest " + name)
        if entry["kind"] not in ARTIFACT_KINDS:
            raise HandoffError("Unknown artifact kind: " + name)

    def require(name, kind=None):
        _relative(name, "artifact reference")
        if name not in artifacts:
            raise HandoffError("Artifact reference lacks a digest: " + name)
        if kind is not None and artifacts[name]["kind"] != kind:
            raise HandoffError(f"Artifact {name} must have kind {kind}")

    runtime = _object(data["runtime"],
                      ("kind", "provenance", "provenance_record", "headers", "crt_objects", "archives", "cxx_abi_archives"),
                      "runtime")
    if runtime["kind"] == "source-built":
        source = _object(runtime["provenance"],
                         ("repository", "revision", "patches_sha256"), "runtime.provenance")
        _repository(source["repository"], "runtime.provenance.repository")
        _revision(source["revision"], "runtime.provenance.revision")
        _hex(source["patches_sha256"], 64, "runtime.provenance.patches_sha256")
    elif runtime["kind"] == "stock-sdk":
        sdk = _object(runtime["provenance"], ("version", "archive_sha256"), "runtime.provenance")
        _text(sdk["version"], "runtime.provenance.version")
        _hex(sdk["archive_sha256"], 64, "runtime.provenance.archive_sha256")
    else:
        raise HandoffError("runtime.kind must be source-built or stock-sdk")
    require(runtime["provenance_record"], "data")
    headers = _object(runtime["headers"], ("path", "tree_sha256"), "runtime.headers")
    _relative(headers["path"], "runtime.headers.path")
    _hex(headers["tree_sha256"], 64, "runtime.headers.tree_sha256")
    for field, kind in (("crt_objects", "object"), ("archives", "archive"), ("cxx_abi_archives", "archive")):
        for name in _list(runtime[field], "runtime." + field):
            require(name, kind)
    toolchain = _object(data["toolchain"], (*TOOLS, "support_files"), "toolchain")
    for field in TOOLS:
        require(toolchain[field], "executable")
    for name in _list(toolchain["support_files"], "toolchain.support_files"):
        require(name)

    identity = _hex(data["runtime_identity_sha256"], 64, "runtime_identity_sha256")
    if identity != runtime_identity_sha256(data):
        raise HandoffError("Runtime identity does not match compiler/header/CRT/C++ ABI inputs")
    components = _object(data["components"], COMPONENTS, "components")
    for name in COMPONENTS:
        component = _object(components[name],
                            ("version", "target_triple", "runtime_identity_sha256",
                             "archives", "support_objects", "provenance"), "components." + name)
        _text(component["version"], name + ".version")
        triple = target["rust_triple" if name in ("rust_std", "webgpu") else "c_triple"]
        if component["target_triple"] != triple:
            raise HandoffError(f"{name}: component target triple differs from the foundation")
        if component["runtime_identity_sha256"] != identity:
            raise HandoffError(f"{name}: component was built for a different declared runtime")
        for archive in _list(component["archives"], name + ".archives"):
            require(archive, "archive")
        for obj in _list(component["support_objects"], name + ".support_objects", nonempty=False):
            require(obj, "object")
        require(component["provenance"], "data")

    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise HandoffError("Artifact root must be a directory")
    for name, entry in artifacts.items():
        path = _file(root, name)
        if _digest(path) != entry["sha256"]:
            raise HandoffError("Artifact SHA256 mismatch: " + name)
        if entry["kind"] == "archive":
            _static_archive(path, name)
        if entry["kind"] == "executable" and not path.stat().st_mode & stat.S_IXUSR:
            raise HandoffError("Tool artifact is not executable: " + name)
    if header_tree_sha256(root, headers["path"]) != headers["tree_sha256"]:
        raise HandoffError("Runtime header tree SHA256 mismatch")
    return {"schema_version": SCHEMA_VERSION, "source_revision": foundation["revision"],
            "runtime_kind": runtime["kind"], "runtime_identity_sha256": identity,
            "target": target, "artifacts_verified": len(artifacts),
            "verification": "static-identity-only", "toolchain_selected": False,
            "runtime_behavior_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", required=True, type=Path, help="materialized artifact export root")
    parser.add_argument("--foundation-repository", required=True)
    parser.add_argument("--foundation-revision", required=True)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"), default="x86_64")
    args = parser.parse_args(argv)
    try:
        result = validate_handoff(load_handoff(args.manifest), args.root,
                                  expected_repository=args.foundation_repository,
                                  expected_revision=args.foundation_revision,
                                  expected_architecture=args.architecture)
    except (HandoffError, OSError) as error:
        print("Foundation handoff rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
