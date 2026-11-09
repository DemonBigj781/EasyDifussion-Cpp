#!/usr/bin/env python3
"""Non-model source consumer contract cases using synthetic local fixtures."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import foundation_consumer as consumer
import foundation_handoff as handoff


REPOSITORY = "https://github.com/example/foundation.git"
REVISION = "0123456789abcdef" * 2 + "01234567"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def archive():
    payload = b"fixture object, never executed"
    header = (b"fixture.o/".ljust(16) + b"0".ljust(12) + b"0".ljust(6) +
              b"0".ljust(6) + b"100644".ljust(8) + str(len(payload)).encode().ljust(10) + b"`\n")
    return b"!<arch>\n" + header + payload + (b"\n" if len(payload) % 2 else b"")


class FoundationConsumerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sdk = self.root / "sdk"
        self.pin = {"repository": REPOSITORY, "revision": REVISION}
        self.patches = {"backend/patches/required.patch": digest(b"application patch")}
        self.data = {
            "format": handoff.FORMAT, "schema_version": 1,
            "foundation": {**self.pin, "source_tree_sha256": digest(b"source"), "clean": True},
            "target": {"architecture": "x86_64", "abi": "cosmopolitan-ape",
                       "c_triple": "x86_64-linux-cosmo", "rust_triple": "x86_64-unknown-linux-musl"},
            "runtime": {"kind": "source-built",
                        "provenance": {**self.pin, "patches_sha256": handoff._canonical_hash([])},
                        "provenance_record": "sdk/SOURCE_RUNTIME.json",
                        "headers": {"path": "sdk/include", "tree_sha256": ""},
                        "crt_objects": ["sdk/x86_64-linux-cosmo/lib/crt.o"],
                        "archives": ["sdk/x86_64-linux-cosmo/lib/libcosmo.a"],
                        "cxx_abi_archives": ["sdk/x86_64-linux-cosmo/lib/libcxx.a"]},
            "toolchain": {}, "components": {}, "artifacts": {}, "runtime_identity_sha256": "",
        }
        names = {"c_compiler": "x86_64-unknown-cosmo-cc", "cxx_compiler": "x86_64-unknown-cosmo-c++",
                 "archiver": "x86_64-unknown-cosmo-ar", "ranlib": "x86_64-linux-cosmo-ranlib",
                 "strip": "x86_64-linux-cosmo-strip", "ape_linker": "apelink", "ape_loader": "ape-x86_64.elf"}
        for role, filename in names.items():
            name = "sdk/bin/" + filename
            self.data["toolchain"][role] = name
            self.artifact(name, "executable", b"#!/bin/sh\nexit 1\n")
        self.artifact("sdk/include/stdio.h", "data", b"/* header fixture */\n")
        self.artifact("sdk/x86_64-linux-cosmo/lib/crt.o", "object", b"fixture CRT")
        for filename in ("libcosmo.a", "libcxx.a", "compiler.a"):
            self.artifact("sdk/x86_64-linux-cosmo/lib/" + filename, "archive", archive())
        self.data["toolchain"]["support_files"] = ["sdk/x86_64-linux-cosmo/lib/compiler.a"]
        self.data["runtime"]["headers"]["tree_sha256"] = handoff.header_tree_sha256(self.root, "sdk/include")
        records = {}
        for name, entry in self.data["artifacts"].items():
            path = self.root / name
            records[path.relative_to(self.sdk).as_posix()] = {
                "type": "file", "mode": path.stat().st_mode & 0o7777,
                "size": path.stat().st_size, "sha256": entry["sha256"],
            }
        source = {
            "schema_version": 1, "kind": "source-built-cosmopolitan-runtime",
            "runtime_abi": "cosmopolitan-source-runtime-v1", "architectures": ["x86_64", "aarch64"],
            "source": {**self.pin, "tree": REVISION, "clean": True},
            "bootstrap": {"role": "stock-compiler-input-only", "version": "4.0.2",
                          "archive_sha256": digest(b"bootstrap"), "tree_sha256": digest(b"bootstrap tree")},
            "source_abi_inputs": {name: digest(name.encode()) for name in
                                  ("libc/nt/struct/sid.h", "libc/dlopen/dladdr.c",
                                   "libc/intrin/fds.h", "libc/proc/posix_spawn.c")},
            "producer": {"entrypoint": "tool/cosmocc/package.sh", "sha256": digest(b"package recipe"),
                         "fresh_outputs": True}, "files": records,
        }
        source_inputs = {
            "schema_version": 1, "kind": "cosmopolitan-tracked-source-inputs", "source": source["source"],
            "files": {"fixture.c": {"type": "file", "mode": "100644", "size": 6, "sha256": digest(b"source")}},
        }
        for name, expected in source["source_abi_inputs"].items():
            source_inputs["files"][name] = {"type": "file", "mode": "100644", "size": len(name), "sha256": expected}
        source_inputs["files"]["tool/cosmocc/package.sh"] = {
            "type": "file", "mode": "100755", "size": len(b"package recipe"), "sha256": digest(b"package recipe"),
        }
        inputs_bytes = json.dumps(source_inputs).encode()
        self.artifact("sdk/SOURCE_INPUTS.json", "data", inputs_bytes)
        records["SOURCE_INPUTS.json"] = {"type": "file", "mode": 0o644, "size": len(inputs_bytes),
                                         "sha256": digest(inputs_bytes)}
        self.data["toolchain"]["support_files"].append("sdk/SOURCE_INPUTS.json")
        source["source_tree_sha256"] = handoff._canonical_hash(source_inputs["files"])
        source["source_inventory_record"] = {"path": "SOURCE_INPUTS.json", "sha256": digest(inputs_bytes)}
        source["source_patches"] = []
        self.data["foundation"]["source_tree_sha256"] = source["source_tree_sha256"]
        source["runtime_identity"] = "sha256:" + handoff._canonical_hash(source)
        self.upstream = {key: source[key] for key in ("kind", "source", "runtime_identity", "runtime_abi")}
        self.artifact("sdk/SOURCE_RUNTIME.json", "data", json.dumps(source).encode())
        for component in handoff.COMPONENTS:
            self.artifact("deps/lib/" + component + ".a", "archive", archive())
            self.artifact("deps/provenance/" + component + ".json", "data",
                          json.dumps({"runtime": self.upstream}).encode())
            self.data["components"][component] = {
                "version": "fixture-1", "runtime_identity_sha256": "",
                "target_triple": self.data["target"]["rust_triple" if component in ("webgpu", "rust_std") else "c_triple"],
                "archives": ["deps/lib/" + component + ".a"], "support_objects": [],
                "provenance": "deps/provenance/" + component + ".json",
            }
        self.artifact("deps/lib/runtime.o", "object", b"fixture runtime object")
        self.data["components"]["rust_std"]["support_objects"] = ["deps/lib/runtime.o"]
        self.data["components"]["rust_std"]["archives"] = ["deps/lib/webgpu.a"]
        self.artifact("deps/provenance/rust_std.json", "data", json.dumps({
            "runtime": self.upstream, "archive_delivery": {
                "mode": "embedded-in-wgpu", "container_sha256": self.data["artifacts"]["deps/lib/webgpu.a"]["sha256"]},
        }).encode())
        self.artifact("deps/include/webgpu.h", "data", b"/* API fixture */")
        self.artifact("deps/runtime/bridge.c", "data", b"/* bridge fixture */")
        self.artifact("deps/licenses/NOTICE", "data", b"fixture notice")
        absolute = lambda name: str(self.root / name)
        self.link = {
            "format": 1, "static": True, "target": "x86_64 Cosmopolitan APE",
            "runtime_shared_libraries": [], "pin": self.pin, "runtime": self.upstream,
            "sdk_root": str(self.sdk), "application_patches": self.patches,
            "software_vulkan": {"runtime": self.upstream, "static": True, "external_vulkan_loader": False,
                                "llvm": {"runtime": self.upstream, "static": True}},
            "runtime_sources": [absolute("deps/runtime/bridge.c")],
            "runtime_objects": [absolute("deps/lib/runtime.o")],
            "runtime_link_options": [], "wgpu_archive": absolute("deps/lib/webgpu.a"),
            "rust_std_linkage": "embedded-in-wgpu",
            "driver_link_args": ["-Wl,--start-group", absolute("deps/lib/mesa.a"),
                                 absolute("deps/lib/llvm.a"), "-Wl,--end-group", "-lcxx"],
            "include_dirs": [str(self.sdk / "include"), absolute("deps/include")],
            "licenses": absolute("deps/licenses"), "license_sha256": {"NOTICE": digest(b"fixture notice")},
            "file_sha256": {absolute(name): entry["sha256"] for name, entry in self.data["artifacts"].items()
                            if name.startswith("deps/")},
        }
        self.persist()

    def artifact(self, name, kind, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o755 if kind == "executable" else 0o644)
        self.data["artifacts"][name] = {"kind": kind, "sha256": digest(content)}

    def persist(self):
        self.artifact("deps/LINK.json", "data", json.dumps(self.link).encode())
        identity = handoff.runtime_identity_sha256(self.data)
        self.data["runtime_identity_sha256"] = identity
        for component in self.data["components"].values():
            component["runtime_identity_sha256"] = identity
        (self.root / "HANDOFF.json").write_text(json.dumps(self.data))

    def select(self):
        return consumer.select_foundation(self.root / "HANDOFF.json", self.root, self.sdk,
                                          self.root / "deps/LINK.json", pin=self.pin,
                                          application_patches=self.patches)

    def make_relative(self):
        relative = lambda path: Path(path).relative_to(self.root).as_posix()
        self.link["path_mode"] = "export-root-relative"
        for field in ("sdk_root", "wgpu_archive", "licenses"):
            self.link[field] = relative(self.link[field])
        for field in ("include_dirs", "runtime_sources", "runtime_objects", "driver_link_args"):
            self.link[field] = [value if value.startswith("-") else relative(value)
                                for value in self.link[field]]
        self.link["file_sha256"] = {relative(path): value for path, value in self.link["file_sha256"].items()}
        self.persist()

    def output_directory(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "generated"

    def reject(self, message):
        with self.assertRaisesRegex(handoff.HandoffError, message):
            self.select()

    def separate_rust(self):
        self.data["components"]["rust_std"]["archives"] = ["deps/lib/rust_std.a"]
        self.link["rust_std_linkage"] = "separate-archives"
        self.artifact("deps/provenance/rust_std.json", "data", json.dumps({
            "runtime": self.upstream, "archive_delivery": {
                "mode": "separate-archives", "archives": {
                    "deps/lib/rust_std.a": self.data["artifacts"]["deps/lib/rust_std.a"]["sha256"]}},
        }).encode())
        self.link["file_sha256"][str(self.root / "deps/provenance/rust_std.json")] = self.data["artifacts"]["deps/provenance/rust_std.json"]["sha256"]

    def test_complete_explicit_export_selects_the_bound_tools(self):
        selected = self.select()
        self.assertEqual(selected["sdk"], self.sdk)
        self.assertEqual(selected["runtime"], self.upstream)
        self.assertEqual(selected["link_sha256"], digest((self.root / "deps/LINK.json").read_bytes()))

    def test_application_patch_is_required(self):
        self.link["application_patches"] = {}
        self.persist()
        self.reject("application patch identity")

    def test_absolute_source_still_gets_a_bound_generated_view(self):
        selected = self.select()
        view = selected["link_view"]
        self.assertEqual(view["path_mode"], "absolute")
        self.assertEqual(view["consumer_view"], {
            "schema_version": 1, "source_manifest_sha256": selected["link_sha256"],
            "source_path_mode": "absolute", "export_root": str(self.root),
        })
        self.assertEqual(selected["link_view_sha256"], digest(consumer.link_view_bytes(view)))

    def test_relative_link_normalizes_only_selected_paths(self):
        original_archive = self.link["wgpu_archive"]
        self.link["software_vulkan"]["producer_path"] = "/producer/original/archive.a"
        self.make_relative()
        before = (self.root / "deps/LINK.json").read_bytes()
        selected = self.select()
        view = selected["link_view"]
        self.assertEqual(view["wgpu_archive"], original_archive)
        self.assertEqual(view["sdk_root"], str(self.sdk))
        self.assertEqual(view["runtime_objects"], [str(self.root / "deps/lib/runtime.o")])
        self.assertEqual(view["runtime_sources"], [str(self.root / "deps/runtime/bridge.c")])
        self.assertEqual(view["include_dirs"], [str(self.sdk / "include"), str(self.root / "deps/include")])
        self.assertIn(str(self.root / "deps/lib/llvm.a"), view["driver_link_args"])
        self.assertIn(str(self.root / "deps/lib/llvm.a"), view["file_sha256"])
        self.assertEqual(view["license_sha256"], self.link["license_sha256"])
        self.assertEqual(view["software_vulkan"], self.link["software_vulkan"])
        self.assertEqual(view["runtime"], self.upstream)
        self.assertEqual((self.root / "deps/LINK.json").read_bytes(), before)
        self.assertEqual(selected["link_sha256"], digest(before))

    def test_relocation_changes_only_consumer_location_binding(self):
        self.make_relative()
        before = self.select()
        relocated = self.output_directory()
        shutil.copytree(self.root, relocated)
        after = consumer.select_foundation(relocated / "HANDOFF.json", relocated, relocated / "sdk",
                                           relocated / "deps/LINK.json", pin=self.pin,
                                           application_patches=self.patches)
        self.assertEqual(before["handoff_sha256"], after["handoff_sha256"])
        self.assertEqual(before["link_sha256"], after["link_sha256"])
        self.assertEqual(before["handoff_report"], after["handoff_report"])
        self.assertNotEqual(before["link_view_sha256"], after["link_view_sha256"])
        self.assertNotEqual(before["selection_identity_sha256"], after["selection_identity_sha256"])
        self.assertEqual(after["link_view"]["sdk_root"], str(relocated / "sdk"))

    def test_relative_mode_cannot_contain_absolute_inputs(self):
        self.make_relative()
        self.link["wgpu_archive"] = str(self.root / "deps/lib/webgpu.a")
        self.persist()
        self.reject("normalized relative POSIX path")

    def test_relative_mode_cannot_escape_or_normalize_parent_components(self):
        self.make_relative()
        self.link["runtime_sources"] = ["deps/runtime/../runtime/bridge.c"]
        self.persist()
        self.reject("normalized relative POSIX path")

    def test_relative_digest_keys_must_also_be_relative(self):
        self.make_relative()
        self.link["file_sha256"][str(self.root / "deps/lib/webgpu.a")] = digest(archive())
        self.persist()
        self.reject("normalized relative POSIX path")

    def test_unknown_path_mode_is_rejected(self):
        self.link["path_mode"] = "auto"
        self.persist()
        self.reject("Unsupported dependency manifest path_mode")

    def test_malformed_relative_path_array_is_rejected(self):
        self.make_relative()
        self.link["include_dirs"] = "deps/include"
        self.persist()
        self.reject("Invalid dependency manifest field: include_dirs")

    def test_generated_view_cannot_be_resubmitted_as_producer_evidence(self):
        self.link["consumer_view"] = {"source_manifest_sha256": digest(b"forged source")}
        self.persist()
        self.reject("original producer LINK.json")

    def test_generated_view_write_is_exact_and_repeatable(self):
        self.make_relative()
        selected = self.select()
        output = self.output_directory()
        path = consumer.write_link_view(output, selected)
        self.assertEqual(digest(path.read_bytes()), selected["link_view_sha256"])
        self.assertEqual(consumer.write_link_view(output, selected), path)
        self.assertEqual(json.loads(path.read_bytes()), selected["link_view"])

    def test_existing_modified_view_is_not_silently_overwritten(self):
        selected = self.select()
        output = self.output_directory()
        path = consumer.write_link_view(output, selected)
        path.write_text("modified")
        with self.assertRaisesRegex(handoff.HandoffError, "Existing generated dependency view"):
            consumer.write_link_view(output, selected)

    def test_in_memory_view_mutation_is_rejected_before_writing(self):
        selected = self.select()
        selected["link_view"]["runtime"] = {"kind": "stock-sdk"}
        output = self.output_directory()
        with self.assertRaisesRegex(handoff.HandoffError, "changed after selection"):
            consumer.write_link_view(output, selected)
        self.assertFalse(output.exists())

    def test_generated_view_cannot_modify_the_export(self):
        selected = self.select()
        with self.assertRaisesRegex(handoff.HandoffError, "outside the immutable export"):
            consumer.write_link_view(self.root / "generated", selected)
        self.assertFalse((self.root / "generated").exists())

    def test_generated_view_cannot_use_a_symlink_destination(self):
        output = self.output_directory()
        output.mkdir()
        (output / "LINK.json").symlink_to(self.root / "deps/LINK.json")
        with self.assertRaisesRegex(handoff.HandoffError, "must be a regular file"):
            consumer.write_link_view(output, self.select())

    def test_stock_sdk_identity_cannot_leak_into_source_link(self):
        self.link["sdk_archive_sha256"] = digest(b"stock SDK")
        self.persist()
        self.reject("does not match the selected source foundation")

    def test_wrong_runtime_component_is_rejected(self):
        self.link["software_vulkan"]["llvm"]["runtime"] = {"kind": "stock-sdk"}
        self.persist()
        self.reject("different source runtime")

    def test_missing_shim_is_not_a_complete_runtime_closure(self):
        self.link["runtime_objects"] = []
        self.persist()
        self.reject("object link list")

    def test_missing_llvm_archive_is_not_a_complete_driver_closure(self):
        self.link["driver_link_args"].remove(str(self.root / "deps/lib/llvm.a"))
        self.persist()
        self.reject("archive link list")

    def test_extra_webgpu_archive_cannot_be_silently_ignored(self):
        self.data["components"]["webgpu"]["archives"].append("deps/lib/rust_std.a")
        self.persist()
        self.reject("entire declared WebGPU archive delivery")

    def test_rust_std_delivery_must_be_explicit(self):
        del self.link["rust_std_linkage"]
        self.persist()
        self.reject("Rust std delivery must be explicit")

    def test_unlinked_standalone_rust_std_is_rejected(self):
        self.separate_rust()
        self.persist()
        self.reject("archive link list")

    def test_separate_rust_std_is_consumed_when_declared(self):
        self.separate_rust()
        self.link["driver_link_args"].append(str(self.root / "deps/lib/rust_std.a"))
        self.persist()
        self.select()

    def test_embedded_rust_std_cannot_name_an_unlinked_separate_archive(self):
        self.data["components"]["rust_std"]["archives"] = ["deps/lib/rust_std.a"]
        self.persist()
        self.reject("Embedded Rust std requires")

    def test_external_library_search_is_rejected(self):
        self.link["driver_link_args"].append("-L/usr/lib")
        self.persist()
        self.reject("Unreviewed source-foundation link option")

    def test_external_include_directory_is_rejected(self):
        self.link["include_dirs"].append("/usr/include")
        self.persist()
        self.reject("outside the verified export")

    def test_cmake_list_brackets_are_rejected(self):
        self.link["include_dirs"].append(str(self.root / "invalid[include"))
        self.persist()
        self.reject("Unrepresentable CMake input path")

    def test_missing_license_inventory_is_rejected(self):
        self.link["license_sha256"] = {}
        self.persist()
        self.reject("license digest inventory")

    def test_updated_link_metadata_changes_the_selection_snapshot(self):
        before = self.select()
        self.link["runtime_link_options"] = ["-Wl,--defsym,__isoc23_strtol=strtol"]
        self.persist()
        self.assertNotEqual(self.select(), before)

    def test_source_sdk_extra_file_is_rejected(self):
        (self.sdk / "untracked-file").write_text("not in producer inventory")
        self.reject("missing or unexpected files")

    def test_handoff_source_tree_must_match_producer_source_inputs(self):
        self.data["foundation"]["source_tree_sha256"] = digest(b"unrelated source")
        self.persist()
        self.reject("differs from the tracked source inventory")

    def test_handoff_source_patch_digest_must_match_committed_source_profile(self):
        self.data["runtime"]["provenance"]["patches_sha256"] = digest(b"unrecorded patches")
        self.persist()
        self.reject("empty post-commit patch list")

    def test_inventory_raw_bytes_are_bound_independently_of_handoff_artifact(self):
        self.artifact("sdk/SOURCE_INPUTS.json", "data", b"{}")
        self.persist()
        self.reject("inventory bytes differ")

    def test_source_inventory_evidence_is_required(self):
        record = json.loads((self.sdk / "SOURCE_RUNTIME.json").read_text())
        del record["source_inventory_record"]
        with self.assertRaisesRegex(handoff.HandoffError, "requires its bound SOURCE_INPUTS.json"):
            consumer.verify_source_inputs(self.sdk, record, self.data)

    def test_abi_input_hashes_must_agree_with_complete_source_inventory(self):
        record = json.loads((self.sdk / "SOURCE_RUNTIME.json").read_text())
        record["source_abi_inputs"]["libc/intrin/fds.h"] = digest(b"different ABI")
        with self.assertRaisesRegex(handoff.HandoffError, "Source ABI input differs"):
            consumer.verify_source_inputs(self.sdk, record, self.data)

    def test_recipe_hash_must_agree_with_complete_source_inventory(self):
        record = json.loads((self.sdk / "SOURCE_RUNTIME.json").read_text())
        record["producer"]["sha256"] = digest(b"different recipe")
        with self.assertRaisesRegex(handoff.HandoffError, "Source package recipe differs"):
            consumer.verify_source_inputs(self.sdk, record, self.data)

    def test_source_runtime_cannot_add_post_commit_patches(self):
        record = json.loads((self.sdk / "SOURCE_RUNTIME.json").read_text())
        record["source_patches"] = ["unrecorded.patch"]
        with self.assertRaisesRegex(handoff.HandoffError, "empty post-commit patch list"):
            consumer.verify_source_inputs(self.sdk, record, self.data)

    def test_source_tool_role_cannot_select_another_sdk_program(self):
        self.data["toolchain"]["c_compiler"] = self.data["toolchain"]["ape_linker"]
        self.persist()
        self.reject("expected x86-64 source SDK entry point")

    def test_simple_alias_materializes_the_recorded_target(self):
        value = {"type": "file", "sha256": digest(b"x"), "size": 1, "mode": 0o644}
        records = {"lib/a": value, "lib/b": {"type": "symlink", "target": "a", "sha256": digest(b"a")}}
        self.assertEqual(consumer.materialized_sdk_files(records), {"lib/a": value, "lib/b": value})

    def test_alias_escape_is_rejected(self):
        records = {"a": {"type": "symlink", "target": "../escape", "sha256": digest(b"../escape")}}
        with self.assertRaisesRegex(handoff.HandoffError, "escapes its root"):
            consumer.materialized_sdk_files(records)

    def test_alias_cycle_is_rejected(self):
        records = {"a": {"type": "symlink", "target": "b", "sha256": digest(b"b")},
                   "b": {"type": "symlink", "target": "a", "sha256": digest(b"a")}}
        with self.assertRaisesRegex(handoff.HandoffError, "Cyclic"):
            consumer.materialized_sdk_files(records)

    def test_directory_alias_materializes_its_complete_subtree(self):
        value = {"type": "file", "sha256": digest(b"x"), "size": 1, "mode": 0o644}
        records = {"directory": {"type": "directory"}, "directory/file": value,
                   "alias": {"type": "symlink", "target": "directory", "sha256": digest(b"directory")}}
        self.assertEqual(consumer.materialized_sdk_files(records), {"directory/file": value, "alias/file": value})

    def test_ancestor_directory_alias_cycle_is_rejected(self):
        records = {"directory": {"type": "directory"},
                   "directory/parent": {"type": "symlink", "target": "..", "sha256": digest(b"..")}}
        with self.assertRaisesRegex(handoff.HandoffError, "Cyclic source SDK directory alias"):
            consumer.materialized_sdk_files(records)

    def test_symlink_expansion_precedes_parent_components(self):
        root_value = {"type": "file", "sha256": digest(b"root"), "size": 4, "mode": 0o644}
        real_value = {"type": "file", "sha256": digest(b"real"), "size": 4, "mode": 0o644}
        records = {
            "file": root_value, "real": {"type": "directory"},
            "real/subdir": {"type": "directory"}, "real/file": real_value,
            "jump": {"type": "symlink", "target": "real/subdir", "sha256": digest(b"real/subdir")},
            "alias": {"type": "symlink", "target": "jump/../file", "sha256": digest(b"jump/../file")},
        }
        files = consumer.materialized_sdk_files(records)
        self.assertEqual(files["alias"], real_value)
        self.assertEqual(files["file"], root_value)

    def test_alias_cannot_walk_through_a_regular_file(self):
        records = {
            "file": {"type": "file", "sha256": digest(b"x"), "size": 1, "mode": 0o644},
            "alias": {"type": "symlink", "target": "file/../file", "sha256": digest(b"file/../file")},
        }
        with self.assertRaisesRegex(handoff.HandoffError, "traverses a file as a directory"):
            consumer.materialized_sdk_files(records)


if __name__ == "__main__":
    unittest.main()
