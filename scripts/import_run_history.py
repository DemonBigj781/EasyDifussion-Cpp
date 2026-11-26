"""Import selected saved renders without reading their embedded metadata."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from easydiffusion.run_history import import_saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--settings", type=Path, help="JSON of verified technical settings; leave unknown fields out")
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else {}
    if not isinstance(settings, dict):
        p.error("--settings must contain a JSON object")
    print(import_saved(args.archive, args.files, settings))


if __name__ == "__main__":
    main()
