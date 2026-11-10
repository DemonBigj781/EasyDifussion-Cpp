import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import audit_cosmo_parity as audit


def git(repo: Path, *args: str) -> None:
    subprocess.check_call(["git", "-C", str(repo), *args])


def write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AuditCosmoParityTests(unittest.TestCase):
    def make_repo(self, root: Path, name: str, files: dict[str, str]) -> Path:
        repo = root / name
        repo.mkdir()
        subprocess.check_call(["git", "init", "-q", str(repo)])
        git(repo, "config", "user.email", "audit@example.invalid")
        git(repo, "config", "user.name", "Audit")
        for relative, text in files.items():
            write(repo, relative, text)
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "seed")
        return repo

    def test_collects_static_parity_without_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = self.make_repo(
                root,
                "main",
                {
                    "shared.txt": "main\n",
                    "main-only.txt": "missing\n",
                    "ui/easydiffusion/server.py": "@router.get('/render')\ndef route(): pass\n",
                },
            )
            theory = self.make_repo(
                root,
                "theory",
                {
                    "shared.txt": "theory\n",
                    "source/DiffUser.cpp/backend.cpp": "// backend\n",
                    "docs/DiffUser.cpp/README.md": "# theory\n",
                    "source/RestAPI.cpp": "CROW_ROUTE(app, '/native')([]{});\n",
                },
            )
            cosmo = self.make_repo(
                root,
                "cosmo",
                {
                    "shared.txt": "cosmo\n",
                    "ui/easydiffusion/online_model_browser/api.py": (
                        "metadata['searchBackend'] = 'model-locations'\n"
                        "metadata['imageListing'] = 'select-model-first'\n"
                    ),
                    "ui/easydiffusion/server.py": "@router.get('/render')\ndef route(): pass\n",
                    "ui/plugins/server/service.py": "@router.post('/plugin')\ndef route(): pass\n",
                    "training/service.py": "print('training')\n",
                },
            )

            result = audit.collect_audit(main, theory, cosmo)

            self.assertEqual(result["file_parity"]["main_missing_in_cosmo"], ["main-only.txt"])
            self.assertEqual(result["python_runtime"]["tracked_production_python_count"], 3)
            self.assertEqual(result["civitai_model_images"]["images_v6_reference_count"], 0)
            self.assertGreaterEqual(
                result["civitai_model_images"]["model_scoped_reference_count"], 2
            )
            self.assertEqual(result["gpu_execution"]["status"], "deferred")
            self.assertEqual(result["routes"]["main"]["count"], 1)
            self.assertEqual(result["routes"]["cosmo"]["count"], 2)

    def test_markdown_output_is_json_safe_for_route_summary(self):
        payload = {
            "repos": {
                name: {
                    "branch": name,
                    "head": "abc1234",
                    "tracked_file_count": 1,
                    "path": f"/{name}",
                }
                for name in ("main", "theory", "cosmo")
            },
            "file_parity": {
                "main_missing_in_cosmo_count": 0,
                "main_missing_in_cosmo": [],
                "theory_missing_in_cosmo_count": 0,
                "theory_missing_in_cosmo_topdir_counts": {},
                "theory_missing_in_cosmo_key_count": 0,
            },
            "python_runtime": {
                "tracked_production_python_count": 0,
                "tracked_production_python_area_counts": {},
            },
            "routes": {
                name: {"count": 1, "by_kind": {"python": 1}}
                for name in ("main", "theory", "cosmo")
            },
            "civitai_model_images": {
                "images_v6_reference_count": 0,
                "model_scoped_reference_count": 1,
            },
        }

        text = audit.to_markdown(payload)

        self.assertIn("# Cosmo parity static audit", text)
        self.assertIn(json.dumps({"python": 1}, sort_keys=True), text)

    def test_text_reader_ignores_directories_and_binary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "tracked-dir").mkdir()
            self.assertEqual(audit.text_for(repo, "tracked-dir"), "")

            binary = repo / "binary.bin"
            binary.write_bytes(b"\xff\xfe\x00")
            self.assertEqual(audit.text_for(repo, "binary.bin"), "")


if __name__ == "__main__":
    unittest.main()
