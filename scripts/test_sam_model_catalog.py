from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sam_folder_fallback_is_available_in_both_model_managers():
    expected = 'SAM_MODEL_FOLDER_NAMES = ("sams", "sam", "SAM")'
    for relative_path in (
        "ui/easydiffusion/model_manager/__init__.py",
        "ui/plugins/server/Browse_files/model_manager/__init__.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in source, relative_path


def test_sam_checkpoint_extension_is_scanned_by_both_model_managers():
    expected = '"sam": [".bin", ".gguf", ".pth"]'
    for relative_path in (
        "ui/easydiffusion/model_manager/__init__.py",
        "ui/plugins/server/Browse_files/model_manager/__init__.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in source, relative_path


if __name__ == "__main__":
    test_sam_folder_fallback_is_available_in_both_model_managers()
    test_sam_checkpoint_extension_is_scanned_by_both_model_managers()
    print("SAM model catalog contract: OK")
