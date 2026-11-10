import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cosmopolitan" / "tests" / "audit_shared_ggml_sources.py"
SPEC = importlib.util.spec_from_file_location("audit_shared_ggml_sources", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def git(repo: Path, *args: str) -> None:
    subprocess.check_call(["git", "-C", str(repo), *args])


def write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SharedGgmlSourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.previous_root = audit.ROOT
        self.previous_pin = audit.PIN
        self.previous_prepare = audit.PREPARE

    def tearDown(self):
        audit.ROOT = self.previous_root
        audit.PIN = self.previous_pin
        audit.PREPARE = self.previous_prepare

    def make_repo(self) -> Path:
        root = Path(tempfile.mkdtemp())
        subprocess.check_call(["git", "init", "-q", str(root)])
        git(root, "config", "user.email", "audit@example.invalid")
        git(root, "config", "user.name", "Audit")
        for relative, text in {
            "source/sd/ggml/include/ggml.h": "#define GGML_MAX_NAME 160\n",
            "source/sd/ggml/include/ggml-backend.h": "typedef int ggml_backend_t;\n",
            "source/sd/ggml/src/ggml.c": "int ggml_init(void) { return 1; }\n",
            "source/sd/ggml/src/ggml.cpp": "int sd_cpp(void) { return 1; }\n",
            "source/llama/ggml/include/ggml.h": "#define GGML_MAX_NAME 128\n",
            "source/llama/ggml/include/ggml-backend.h": "typedef int ggml_backend_t;\n",
            "source/llama/ggml/src/ggml.c": "int ggml_init(void) { return 2; }\n",
            "source/llama/ggml/src/ggml.cpp": "int llama_cpp(void) { return 1; }\n",
            "cosmopolitan/shared-ggml/patches/0001.patch": "diff --git a/x b/x\n",
            "cosmopolitan/shared-ggml/prepare.py": (
                "GGML_MAX_NAME=160\nGGML_COSMO_STATIC_ONLY\n"
                "llama_donor_files\nauthoritative_tree_sha256\n"
            ),
        }.items():
            write(root, relative, text)
        git(root, "add", ".")
        git(root, "commit", "-qm", "seed")
        audit.ROOT = root
        audit.PIN = root / "cosmopolitan" / "shared-ggml" / "PIN.json"
        audit.PREPARE = root / "cosmopolitan" / "shared-ggml" / "prepare.py"
        return root

    def write_pin(self, root: Path, *, include_unexpected=False) -> None:
        authoritative = "source/sd/ggml"
        llama = "source/llama/ggml"
        tree = audit.tree_summary(authoritative)
        donor = {
            relative: audit.digest_file(f"{llama}/{relative}")
            for relative in ("include/ggml.h", "include/ggml-backend.h", "src/ggml.c")
        }
        pin = {
            "schema": 1,
            "authoritative_source": authoritative,
            "authoritative_ggml_base_revision": "sd-base",
            "authoritative_tree_sha256": tree["tree_sha256"],
            "authoritative_file_count": tree["tracked_file_count"],
            "llama_source": llama,
            "llama_revision": "llama-base",
            "ggml_max_name": 160,
            "llama_donor_files": donor,
            "patches": {
                "0001.patch": audit.digest_file("cosmopolitan/shared-ggml/patches/0001.patch"),
            },
        }
        audit.PIN.write_text(__import__("json").dumps(pin, indent=2) + "\n", encoding="utf-8")
        git(root, "add", str(audit.PIN.relative_to(root)))
        if include_unexpected:
            write(root, "third_party/other/ggml/src/ggml.c", "int other(void) { return 0; }\n")
            git(root, "add", "third_party/other/ggml/src/ggml.c")
        git(root, "commit", "-qm", "pin")

    def test_collects_and_validates_expected_shared_ggml_contract(self):
        root = self.make_repo()
        self.write_pin(root)

        result = audit.collect_audit()
        errors = audit.validate(result)

        self.assertEqual(errors, [])
        self.assertEqual(result["trees"]["authoritative"]["matches_pin"], True)
        self.assertEqual(result["ggml_roots"]["unexpected"], [])
        self.assertEqual(result["gpu_execution"]["status"], "deferred")
        different = {item["path"] for item in result["authoritative_vs_llama"] if item["status"] == "different"}
        self.assertIn("include/ggml.h", different)
        self.assertIn("src/ggml.c", different)

    def test_validation_reports_unexpected_extra_ggml_root(self):
        root = self.make_repo()
        self.write_pin(root, include_unexpected=True)

        result = audit.collect_audit()
        errors = audit.validate(result)

        self.assertTrue(any("unexpected tracked GGML roots" in error for error in errors))
        self.assertIn("third_party/other/ggml", result["ggml_roots"]["unexpected"])


if __name__ == "__main__":
    unittest.main()
