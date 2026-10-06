"""Read-only diagnostics rooted at the application directory."""

import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def read_log_tail(path, lines=500):
    if not 1 <= lines <= 1000:
        raise ValueError("Log line count must be between 1 and 1000")
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        start = max(0, size - 512 * 1024)
        stream.seek(start)
        data = stream.read(512 * 1024)
    if start:
        data = data.partition(b"\n")[2]
    entries = data.decode("utf-8", errors="replace").splitlines()
    return {"lines": entries[-lines:], "count": min(len(entries), lines),
            "truncated": bool(start or len(entries) > lines)}


def _metadata(path):
    link = path.is_symlink()
    try:
        info = path.stat()
        kind = "directory" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "special"
    except FileNotFoundError:
        if not link:
            raise
        info = path.lstat()
        kind = "broken symlink"
    result = {"name": path.name, "type": kind, "symlink": link, "size": info.st_size,
              "modified": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
              "permissions": stat.filemode(info.st_mode)}
    if link:
        result["target"] = os.readlink(path)
    return result


def inspect_path(root, path=".", offset=0):
    if offset < 0:
        raise ValueError("Offset must not be negative")
    relative = os.path.normpath(path or ".")
    if os.path.isabs(relative) or relative == ".." or relative.startswith("../") or "\x00" in relative:
        raise ValueError("Use a path relative to the application root; symlinks are allowed")
    target = Path(root) / relative
    result = {**_metadata(target), "path": relative}
    if result["type"] == "directory":
        entries = []
        more = False
        with os.scandir(target) as iterator:
            for index, entry in enumerate(iterator):
                if index < offset:
                    continue
                if len(entries) == 200:
                    more = True
                    break
                try:
                    entries.append(_metadata(Path(entry.path)))
                except OSError:
                    entries.append({"name": entry.name, "type": "unavailable"})
        result["entries"] = sorted(entries, key=lambda entry: (entry["type"] != "directory", entry["name"].casefold()))
        result["next_offset"] = offset + len(entries) if more else None
    return result
