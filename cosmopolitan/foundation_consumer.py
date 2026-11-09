"""Select an explicitly supplied, complete source-runtime application closure.

No source pin is changed and no producer/build is launched by this module.
Consumers must keep the export quiescent until compilation/package completion.
"""

import hashlib
import os
from pathlib import Path
import posixpath
import re
import stat
import sys

from foundation_handoff import (HandoffError, TOOLS, _canonical_hash, _digest,
                                _file, _hex, _relative, _revision, load_handoff,
                                validate_handoff)


LINK_OPTIONS = {
    "-Wl,--start-group", "-Wl,--end-group", "-Wl,--whole-archive",
    "-Wl,--no-whole-archive", "-lcxx", "-lrt", "-ldl", "-lm", "-lpthread",
}


def link_option(value):
    if not isinstance(value, str) or (
            value not in LINK_OPTIONS and
            re.fullmatch(r"-Wl,--defsym,[A-Za-z_][A-Za-z0-9_]*=[A-Za-z_][A-Za-z0-9_]*", value) is None):
        raise HandoffError("Unreviewed source-foundation link option: " + repr(value))
    return value


def _files(directory):
    """Return all regular files, failing on symlinks or enumeration errors."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise HandoffError("Expected a materialized export directory: " + str(directory))
    result = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            path = Path(entry.path)
            if stat.S_ISDIR(mode):
                result.update(_files(path))
            elif stat.S_ISREG(mode):
                result.add(path)
            else:
                raise HandoffError("Export contains a symlink or special file: " + str(path))
    return result


def materialized_sdk_files(records):
    """Resolve file aliases recorded by the producer before materialization.

    Directory aliases are expanded using the recorded tree, not the host's
    filesystem. Cyclic/dangling/escaping aliases fail rather than dropping
    compiler support files from the exported closure.
    """
    if not isinstance(records, dict) or not records:
        raise HandoffError("Source SDK provenance has no file inventory")
    for name, entry in records.items():
        _relative(name, "source SDK path")
        if not isinstance(entry, dict) or entry.get("type") not in ("file", "directory", "symlink"):
            raise HandoffError("Unknown source SDK inventory entry: " + name)
        parts = name.split("/")
        for index in range(1, len(parts)):
            parent = records.get("/".join(parts[:index]))
            if isinstance(parent, dict) and parent.get("type") in ("file", "symlink"):
                raise HandoffError("Source SDK inventory descends through a non-directory: " + name)

    def children(name):
        prefix = name + "/" if name else ""
        return sorted({path[len(prefix):].split("/")[0] for path in records
                       if path.startswith(prefix) and path != name})

    def resolve(name):
        pending = name.split("/") if name else []
        resolved = []
        followed = 0
        while pending:
            part = pending.pop(0)
            if part in ("", "."):
                continue
            if part == "..":
                if not resolved:
                    raise HandoffError("Source SDK alias escapes its root")
                resolved.pop()
                continue
            candidate = "/".join([*resolved, part])
            _relative(candidate, "source SDK alias target")
            entry = records.get(candidate)
            if entry is None and children(candidate):
                entry = {"type": "directory"}
            if entry is None:
                raise HandoffError("Dangling source SDK alias: " + candidate)
            if entry["type"] == "symlink":
                followed += 1
                if followed > 40:
                    raise HandoffError("Cyclic or excessive source SDK alias chain: " + candidate)
                target = entry.get("target")
                if not isinstance(target, str) or not target or "\\" in target or target.startswith("/"):
                    raise HandoffError("Unsafe source SDK alias: " + candidate)
                if entry.get("sha256") != hashlib.sha256(target.encode()).hexdigest():
                    raise HandoffError("Source SDK alias digest mismatch: " + candidate)
                # Expand each symlink before processing later '..' components.
                # Lexical normpath first would select the wrong file for
                # targets such as jump/../file where jump is itself a link.
                pending = target.split("/") + pending
                continue
            resolved.append(part)
            if entry["type"] == "file" and pending:
                raise HandoffError("Source SDK alias traverses a file as a directory: " + candidate)
        name = "/".join(resolved)
        entry = records.get(name, {"type": "directory"})
        if entry["type"] == "file":
            _hex(entry.get("sha256"), 64, "source SDK file digest")
            if type(entry.get("mode")) is not int or type(entry.get("size")) is not int or entry["size"] < 0:
                raise HandoffError("Invalid source SDK file mode/size: " + name)
        return name, entry

    materialized = {}

    def expand(name, destination, ancestors):
        canonical, entry = resolve(name)
        if entry["type"] == "file":
            materialized[destination] = entry
            return
        if canonical in ancestors:
            raise HandoffError("Cyclic source SDK directory alias: " + name)
        for child in children(canonical):
            expand(posixpath.join(canonical, child), posixpath.join(destination, child),
                   ancestors | {canonical})

    expand("", "", set())
    return materialized


def verify_source_sdk(sdk, record_path, handoff):
    """Match materialized SDK bytes to the original source producer record."""
    if record_path != sdk / "SOURCE_RUNTIME.json":
        raise HandoffError("The selected SDK must contain the bound SOURCE_RUNTIME.json")
    record = load_handoff(record_path)
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int or
            record["schema_version"] != 1 or record.get("kind") != "source-built-cosmopolitan-runtime" or
            record.get("runtime_abi") != "cosmopolitan-source-runtime-v1" or
            record.get("architectures") != ["x86_64", "aarch64"]):
        raise HandoffError("Unsupported source runtime producer record")
    identity = "sha256:" + _canonical_hash({key: value for key, value in record.items() if key != "runtime_identity"})
    if record.get("runtime_identity") != identity:
        raise HandoffError("Upstream source runtime identity mismatch")
    source = record.get("source")
    if not isinstance(source, dict) or source.get("clean") is not True:
        raise HandoffError("Source runtime requires clean pinned source")
    _revision(source.get("tree"), "source runtime tree")
    provenance = handoff["runtime"]["provenance"]
    if source.get("repository") != provenance["repository"] or source.get("revision") != provenance["revision"]:
        raise HandoffError("Source runtime record differs from selected runtime provenance")
    bootstrap = record.get("bootstrap", {})
    producer = record.get("producer", {})
    abi = record.get("source_abi_inputs", {})
    if (not isinstance(bootstrap, dict) or bootstrap.get("role") != "stock-compiler-input-only" or
            not isinstance(producer, dict) or producer.get("fresh_outputs") is not True or
            producer.get("entrypoint") != "tool/cosmocc/package.sh"):
        raise HandoffError("Source runtime does not record a complete fresh package build")
    for field in ("archive_sha256", "tree_sha256"):
        _hex(bootstrap.get(field), 64, "bootstrap." + field)
    required_abi = {"libc/nt/struct/sid.h", "libc/dlopen/dladdr.c",
                    "libc/intrin/fds.h", "libc/proc/posix_spawn.c"}
    if not isinstance(abi, dict) or set(abi) != required_abi:
        raise HandoffError("Source runtime ABI input inventory is incomplete")
    for value in abi.values():
        _hex(value, 64, "source ABI digest")
    expected = materialized_sdk_files(record.get("files"))
    actual = {path.relative_to(sdk).as_posix() for path in _files(sdk)}
    if actual != set(expected) | {"SOURCE_RUNTIME.json"}:
        raise HandoffError("Materialized source SDK has missing or unexpected files")
    for name, entry in expected.items():
        path = _file(sdk, name)
        metadata = path.stat()
        if (metadata.st_size != entry["size"] or stat.S_IMODE(metadata.st_mode) != entry["mode"] or
                _digest(path) != entry["sha256"]):
            raise HandoffError("Materialized source SDK differs from producer input: " + name)
    return {key: record[key] for key in ("kind", "source", "runtime_identity", "runtime_abi")}


def select_foundation(manifest_path, export_root, sdk_path, link_path, *, pin, application_patches):
    """Validate the selected inputs and return only verified build paths."""
    root = Path(export_root).resolve(strict=True)
    data = load_handoff(manifest_path)
    report = validate_handoff(data, root, expected_repository=pin["repository"],
                              expected_revision=pin["revision"], expected_architecture="x86_64")
    if data["runtime"]["kind"] != "source-built":
        raise HandoffError("The source handoff path cannot select stock-SDK inputs")

    def relative(path):
        path = Path(path)
        if not path.is_absolute():
            raise HandoffError("Prebuilt LINK.json paths must be absolute: " + str(path))
        # CMake's JSON-to-list bridge cannot safely preserve these characters.
        if any(character in str(path) for character in (";", "\n", "\r", "$", "[", "]")):
            raise HandoffError("Unrepresentable CMake input path: " + str(path))
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as error:
            raise HandoffError("Build input is outside the verified export: " + str(path)) from error
        _relative(name, "build input")
        return name

    def artifact(path, kind=None):
        name = relative(path)
        if name not in data["artifacts"] or (kind and data["artifacts"][name]["kind"] != kind):
            raise HandoffError("Build input lacks a matching handoff artifact: " + name)
        return _file(root, name)

    sdk = Path(sdk_path).absolute()
    relative(sdk)
    provenance_path = artifact(root / data["runtime"]["provenance_record"], "data")
    upstream_runtime = verify_source_sdk(sdk, provenance_path, data)
    if root / data["runtime"]["headers"]["path"] != sdk / "include":
        raise HandoffError("Runtime headers do not belong to the selected source SDK")
    tools = {name: artifact(root / data["toolchain"][name], "executable") for name in TOOLS}
    tool_names = {"c_compiler": "x86_64-unknown-cosmo-cc", "cxx_compiler": "x86_64-unknown-cosmo-c++",
                  "archiver": "x86_64-unknown-cosmo-ar", "ranlib": "x86_64-linux-cosmo-ranlib",
                  "strip": "x86_64-linux-cosmo-strip", "ape_linker": "apelink", "ape_loader": "ape-x86_64.elf"}
    for name, path in tools.items():
        if path != sdk / "bin" / tool_names[name]:
            raise HandoffError("Tool must select the expected x86-64 source SDK entry point: " + name)
    for field, filename in (("crt_objects", "crt.o"), ("archives", "libcosmo.a"),
                            ("cxx_abi_archives", "libcxx.a")):
        selected_path = relative(sdk / "x86_64-linux-cosmo/lib" / filename)
        if selected_path not in data["runtime"][field]:
            raise HandoffError("Handoff omits the selected x86-64 default runtime input: " + filename)
        for name in data["runtime"][field]:
            if sdk not in (root / name).parents:
                raise HandoffError("Declared runtime input is outside the selected source SDK: " + name)

    link_file = artifact(Path(link_path).absolute(), "data")
    link = load_handoff(link_file)
    if (not isinstance(link, dict) or type(link.get("format")) is not int or link["format"] != 1 or
            link.get("static") is not True or link.get("target") != "x86_64 Cosmopolitan APE" or
            link.get("runtime_shared_libraries") != [] or link.get("pin") != pin or
            link.get("sdk_root") != str(sdk) or link.get("runtime") != upstream_runtime or
            link.get("sdk_archive_sha256") is not None):
        raise HandoffError("Application dependency manifest does not match the selected source foundation")
    if link.get("application_patches") != application_patches:
        raise HandoffError("Prebuilt WGPU lacks the exact EasyDiffusion application patch identity")
    mesa = link.get("software_vulkan", {})
    llvm = mesa.get("llvm", {}) if isinstance(mesa, dict) else {}
    if (not isinstance(mesa, dict) or not isinstance(llvm, dict) or
            mesa.get("runtime") != upstream_runtime or llvm.get("runtime") != upstream_runtime or
            mesa.get("static") is not True or llvm.get("static") is not True or
            mesa.get("external_vulkan_loader") is not False):
        raise HandoffError("Mesa/LLVM static closure belongs to a different source runtime")
    component_evidence = {}
    for name, component in data["components"].items():
        evidence = load_handoff(artifact(root / component["provenance"], "data"))
        if not isinstance(evidence, dict) or evidence.get("runtime") != upstream_runtime:
            raise HandoffError("Component provenance does not bind the selected source runtime")
        component_evidence[name] = evidence

    def sequence(field, nonempty=True):
        values = link.get(field)
        if (not isinstance(values, list) or (nonempty and not values) or
                not all(isinstance(value, str) and value for value in values)):
            raise HandoffError("Invalid dependency manifest field: " + field)
        return values

    for source in sequence("runtime_sources"):
        artifact(source, "data")
    objects = sequence("runtime_objects", nonempty=False)
    for obj in objects:
        artifact(obj, "object")
    expected_objects = {name for component in data["components"].values() for name in component["support_objects"]}
    if {relative(obj) for obj in objects} != expected_objects:
        raise HandoffError("Runtime object link list does not match the component closure")
    wgpu = artifact(link.get("wgpu_archive", ""), "archive")
    wgpu_name = relative(wgpu)
    if data["components"]["webgpu"]["archives"] != [wgpu_name]:
        raise HandoffError("The entire declared WebGPU archive delivery must be the selected staticlib")
    rust_archives = set(data["components"]["rust_std"]["archives"])
    rust_linkage = link.get("rust_std_linkage")
    if rust_linkage == "embedded-in-wgpu":
        # This is an explicit producer receipt, not a claim that this consumer
        # can prove Rust codegen or archive-member equivalence after LTO.
        receipt = {"mode": "embedded-in-wgpu", "container_sha256": data["artifacts"][wgpu_name]["sha256"]}
        if (rust_archives != {wgpu_name} or
                component_evidence["rust_std"].get("archive_delivery") != receipt):
            raise HandoffError("Embedded Rust std requires the selected WGPU container and its bound delivery receipt")
        separate_rust_archives = set()
    elif rust_linkage == "separate-archives":
        if wgpu_name in rust_archives:
            raise HandoffError("Separately linked Rust std cannot name the WGPU container")
        receipt = {"mode": "separate-archives",
                   "archives": {name: data["artifacts"][name]["sha256"] for name in sorted(rust_archives)}}
        if component_evidence["rust_std"].get("archive_delivery") != receipt:
            raise HandoffError("Separate Rust std archives require a matching bound delivery receipt")
        separate_rust_archives = rust_archives
    else:
        raise HandoffError("Rust std delivery must be explicit: embedded-in-wgpu or separate-archives")
    driver_archives = set()
    for argument in sequence("driver_link_args"):
        if argument.startswith("-"):
            link_option(argument)
        else:
            artifact(argument, "archive")
            driver_archives.add(relative(argument))
    expected_archives = (set(data["components"]["mesa"]["archives"] + data["components"]["llvm"]["archives"]) |
                         separate_rust_archives)
    if driver_archives != expected_archives:
        raise HandoffError("Driver/Rust archive link list does not match the declared delivery closure")
    for option in sequence("runtime_link_options", nonempty=False):
        link_option(option)

    def directory(path):
        name = relative(path)
        current = root
        for part in Path(name).parts:
            current /= part
            if current.is_symlink():
                raise HandoffError("Foundation directory cannot contain a symlink: " + str(path))
        files = _files(Path(path))
        if not files:
            raise HandoffError("Empty foundation include/license directory: " + str(path))
        for file in files:
            # The source SDK inventory was checked in full above, including
            # aliases materialized into ordinary files. Other trees need
            # individual handoff artifacts, not unchecked include search paths.
            if sdk not in file.parents:
                artifact(file)
        return files

    for include in sequence("include_dirs"):
        directory(include)
    licenses = Path(link.get("licenses", ""))
    license_files = directory(licenses)
    license_hashes = link.get("license_sha256")
    if not isinstance(license_hashes, dict) or not license_hashes:
        raise HandoffError("Dependency license digest inventory is missing")
    if {path.relative_to(licenses).as_posix() for path in license_files} != set(license_hashes):
        raise HandoffError("Dependency license inventory is incomplete")
    for name, expected in license_hashes.items():
        _relative(name, "license path")
        key = relative(artifact(licenses / name))
        if data["artifacts"][key]["sha256"] != expected:
            raise HandoffError("Dependency license hash differs from handoff: " + name)
    hashes = link.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise HandoffError("Prebuilt dependency file digest inventory is missing")
    for path, expected in hashes.items():
        name = relative(artifact(path))
        if expected != data["artifacts"][name]["sha256"]:
            raise HandoffError("Prebuilt dependency digest differs from handoff: " + name)
    handoff_hash = _digest(Path(manifest_path))
    link_hash = _digest(link_file)
    selection_identity = _canonical_hash({"handoff_sha256": handoff_hash, "link_sha256": link_hash,
                                           "runtime_identity_sha256": report["runtime_identity_sha256"]})
    return {"sdk": sdk, "tools": tools, "link_manifest": link_file,
            "handoff_manifest": Path(manifest_path).resolve(strict=True),
            "handoff_sha256": handoff_hash, "link_sha256": link_hash,
            "selection_identity_sha256": selection_identity,
            "handoff_report": report, "runtime": upstream_runtime}


def tool_command(binary, loader):
    """Select the loader by file format, not a filename convention."""
    with Path(binary).open("rb") as source:
        portable = source.read(2) == b"MZ"
    return [str(loader), str(binary)] if portable else [str(binary)]


def write_wrappers(directory, selection):
    """Create direct source-SDK wrappers without a stock-driver text patch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = {"cc": "c_compiler", "cxx": "cxx_compiler", "ar": "archiver",
             "ranlib": "ranlib", "strip": "strip"}
    for name, field in names.items():
        command = tool_command(selection["tools"][field], selection["tools"]["ape_loader"])
        content = ("#!" + sys.executable + "\nimport os, sys\n" +
                   "command = " + repr(command) + "\n" +
                   "os.execv(command[0], command + sys.argv[1:])\n")
        path = directory / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
        path.chmod(0o755)
    return directory
