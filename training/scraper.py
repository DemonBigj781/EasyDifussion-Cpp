"""Grabber CLI search plus isolated scrape-attempt review galleries."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import re
import socket
import subprocess
import tempfile
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageOps

DEFAULT_CLI = "/home/jack/bin/grabber/Grabber-cli"
SOURCE_BLACKLIST = Path(__file__).with_name(".grabber-source-blacklist")
MAX_RESULTS = 300
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_GALLERY_PNGS = 10000
MOTION_EXTENSIONS = {".gif", ".mp4", ".webm", ".mov", ".mkv", ".avi"}
def cli_path() -> Path:
    return Path(os.environ.get("GRABBER_CLI", DEFAULT_CLI)).expanduser()


def _run(args: list[str], timeout: int = 120) -> str:
    binary = cli_path()
    if not binary.is_file():
        raise ValueError(f"Grabber CLI not found: {binary}; set GRABBER_CLI")
    try:
        result = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Grabber search timed out after {timeout} seconds") from exc
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or "Grabber failed")[-3000:])
    return result.stdout


def sources() -> list[str]:
    """List installed Grabber sources, excluding locally blacklisted adapters."""
    blocked = {
        line.strip().casefold()
        for line in SOURCE_BLACKLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    } if SOURCE_BLACKLIST.is_file() else set()
    found = {line.strip() for line in _run(["source", "list"]).splitlines() if line.strip()}
    return sorted(site for site in found if site.casefold() not in blocked)


def tag_suggestions(query: str, selected_sources: list[str]) -> list[str]:
    term = query.strip()
    if len(term) < 2 or len(term) > 100:
        return []
    available = set(sources())
    if len(selected_sources) > 50 or any(site not in available for site in selected_sources):
        raise ValueError("Choose websites from Grabber's source list")
    selected_sources = selected_sources or sorted(available)
    output = _run(["--ignore-error", "--tags", term, "--sources", " ".join(selected_sources),
                   "--tags-format", "%tag", "--tags-min", "1", "--return-tags"], timeout=45)
    suggestions = []
    seen = set()
    for line in output.splitlines():
        tag = line.split("\t", 1)[0].strip()
        key = tag.casefold()
        if tag and term.casefold() in key and key not in seen:
            suggestions.append(tag)
            seen.add(key)
            if len(suggestions) >= 20:
                break
    return suggestions


def _attempt_folder(dataset_root: Path, dataset: str, attempt_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", attempt_id):
        raise ValueError("Invalid scrape attempt id")
    root = dataset_root.resolve()
    folder = (root / dataset).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("Dataset path leaves the configured dataset directory")
    scrape_root = folder / ".scrapes"
    if scrape_root.is_symlink():
        raise ValueError("Scrape gallery directories must not be symlinks")
    attempt = scrape_root / attempt_id
    if attempt.is_symlink() or not attempt.resolve().is_relative_to(folder):
        raise ValueError("Scrape attempt path is invalid")
    for child in (attempt / "before", attempt / "filtered", attempt / "frames"):
        if child.is_symlink() or (child.exists() and not child.resolve().is_relative_to(attempt.resolve())):
            raise ValueError("Scrape gallery subfolders must not be symlinks")
    return attempt


def _manifest(folder: Path) -> dict:
    path = folder / "attempt.json"
    if folder.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Scrape attempt does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def search(dataset_root: Path, dataset: str, query: str, selected_sources: list[str], limit: int) -> dict:
    if not query.strip() or len(query) > 500:
        raise ValueError("Enter a search query up to 500 characters")
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"Choose between 1 and {MAX_RESULTS} search results")
    available = set(sources())
    if len(selected_sources) > 50 or any(s not in available for s in selected_sources):
        raise ValueError("Choose websites from Grabber's source list")
    selected_sources = selected_sources or sorted(available)
    output = _run(["--ignore-error", "--return-images", "--json", "--load-details", "--tags", query.strip(), "--sources",
                   " ".join(selected_sources), "--page", "1", "--perpage", str(min(limit, 100)),
                   "--max", str(limit), "--no-duplicates"], timeout=180)
    try:
        raw_items = json.loads(output)
    except json.JSONDecodeError as exc:
        # Some Grabber builds emit the documented one-URL-per-line format even
        # when JSON output is requested.
        raw_items = re.findall(r"https?://[^\s\"'<>]+", output)
        if not raw_items:
            raise ValueError(f"Grabber returned invalid search results: {output[-1000:]}") from exc
    if isinstance(raw_items, dict):
        wrapped = raw_items.get("images", raw_items.get("results", raw_items.get("items", raw_items.get("data"))))
        raw_items = wrapped if wrapped is not None else ([raw_items] if any(
            key in raw_items for key in ("url", "url_file", "url_original", "file_url")) else [])
        if isinstance(raw_items, dict):
            raw_items = list(raw_items.values())
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raise ValueError("Grabber returned an unexpected search result")
    reported_count = len(raw_items)
    attempt_id = uuid.uuid4().hex[:12]
    gallery = _attempt_folder(dataset_root, dataset, attempt_id)
    (gallery / "before").mkdir(parents=True, exist_ok=False)
    (gallery / "filtered").mkdir()
    (gallery / "frames").mkdir()
    results = []
    for raw in raw_items[:limit]:
        if isinstance(raw, str):
            raw = {"url": raw}
        if not isinstance(raw, dict):
            continue
        data = raw.get("data", {}) if isinstance(raw.get("data"), dict) else {}
        sizes = raw.get("sizes", {}) if isinstance(raw.get("sizes"), dict) else {}
        full_size = sizes.get("full", {}) if isinstance(sizes.get("full", {}), dict) else {}
        thumb_size = sizes.get("thumbnail", {}) if isinstance(sizes.get("thumbnail", {}), dict) else {}
        page_url = raw.get("page_url", raw.get("url_page", data.get("page_url", "")))
        # Grabber's normalized Image fields are file_url, sample_url, and
        # preview_url. Keep accepting URL aliases for older/custom sources.
        url = (raw.get("file_url") or raw.get("url_file") or raw.get("url_original")
               or full_size.get("url") or raw.get("url") or data.get("file_url"))
        preview_url = (raw.get("preview_url") or raw.get("url_thumbnail") or raw.get("thumbnail_url")
                       or raw.get("sample_url") or raw.get("url_sample") or thumb_size.get("url")
                       or data.get("preview_url") or data.get("sample_url") or url)
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(preview_url, dict):
            preview_url = preview_url.get("url")
        base_url = page_url if isinstance(page_url, str) and urllib.parse.urlparse(page_url).scheme in {"https", "http"} else ""
        if not base_url and isinstance(url, str) and urllib.parse.urlparse(url).scheme in {"https", "http"}:
            base_url = url
        if isinstance(url, str):
            url = urllib.parse.urljoin(base_url, url)
        if isinstance(preview_url, str):
            preview_url = urllib.parse.urljoin(url if isinstance(url, str) else base_url, preview_url)
        if not isinstance(url, str) or urllib.parse.urlparse(url).scheme not in {"https", "http"}:
            continue
        if not isinstance(preview_url, str) or urllib.parse.urlparse(preview_url).scheme not in {"https", "http"}:
            preview_url = url
        key = hashlib.sha256((url + "\n" + str(raw.get("id", ""))).encode()).hexdigest()[:20]
        raw_tags = (raw.get("tags") or data.get("tags") or raw.get("all_tags") or raw.get("all")
                    or data.get("all_tags") or data.get("all") or raw.get("tag_string") or "")
        if isinstance(raw_tags, dict):
            raw_tags = [tag for values in raw_tags.values()
                        for tag in (values if isinstance(values, list) else [values])]
        if isinstance(raw_tags, str):
            tags = [tag.strip() for tag in re.split(r"[,\t\n ]+", raw_tags) if tag.strip()]
        elif isinstance(raw_tags, list):
            tags = []
            for value in raw_tags:
                if isinstance(value, dict):
                    value = value.get("text", value.get("name", value.get("tag", "")))
                if value:
                    tags.append(str(value).strip())
        else:
            tags = []
        extension = Path(urllib.parse.urlparse(url).path).suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            extension = ".bin"
        results.append({"key": key, "url": url, "preview_url": preview_url,
                        "extension": extension, "page_url": str(page_url),
                        "source": str(raw.get("website", raw.get("source", " ".join(selected_sources)))),
                        "md5": str(raw.get("md5", "")), "name": str(raw.get("name", "")),
                        "tags": tags[:200] if isinstance(tags, list) else []})
    data = {"id": attempt_id, "dataset": dataset, "query": query.strip(),
            "sources": selected_sources, "results": results, "files": [], "frame_files": [], "before_files": [],
            "threshold": .95, "status": "searched"}
    (gallery / "attempt.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"attempt": attempt_id, "reported_count": reported_count, "results": [{k: v for k, v in item.items() if k != "url"}
                                                   | {"url": item.get("preview_url", item["url"])}
                                                   for item in results], "count": len(results)}


def attempts(dataset_root: Path, dataset: str) -> list[dict]:
    root = (dataset_root / dataset).resolve() / ".scrapes"
    if root.is_symlink():
        raise ValueError("Scrape gallery directories must not be symlinks")
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            item = _manifest(folder)
            found.append({"attempt": item["id"], "query": item["query"],
                          "sources": item["sources"], "status": item["status"],
                          "images": len(item.get("files", [])) + len(item.get("frame_files", []))})
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return found


def _public_web_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Only HTTP or HTTPS image URLs are accepted")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or default_port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("Could not resolve image host") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Image host must resolve to a public IP address")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_web_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, destination: Path) -> None:
    _public_web_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "EasyDiffusion-GrabberGallery/1.0"})
    opener = urllib.request.build_opener(_CheckedRedirects())
    with opener.open(request, timeout=45) as response, destination.open("wb") as output:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Downloaded file exceeds 100 MiB")
            output.write(chunk)


def _resize(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image


def _dhash_image(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        return _dhash_image(image)


def _png_name(image: Image.Image) -> tuple[str, bytes, int]:
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as output:
        image.save(output, format="PNG", optimize=True)
        output.seek(0)
        data = output.read()
    width, height = image.size
    digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
    return f"{digest}_{width}x{height}.png", data, int(digest, 16)


def _frames(source: Path):
    suffix = source.suffix.lower()
    if suffix not in MOTION_EXTENSIONS:
        with Image.open(source) as opened:
            yield _resize(opened)
        return
    if suffix == ".gif":
        with Image.open(source) as opened:
            frame_count = getattr(opened, "n_frames", 1)
            for index in range(0, frame_count, 60):
                opened.seek(index)
                yield _resize(opened.copy())
        return
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("Video conversion needs ffmpeg and ffprobe")
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate,duration,nb_frames", "-of", "json", str(source)],
                           capture_output=True, text=True, timeout=20)
    if probe.returncode:
        raise ValueError("Could not inspect video")
    if not json.loads(probe.stdout).get("streams"):
        raise ValueError("Video has no readable video stream")
    frames_dir = Path(tempfile.mkdtemp(prefix="grabber-frames-"))
    try:
        subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-vf", r"select=not(mod(n\,60))",
                        "-fps_mode", "passthrough", str(frames_dir / "frame-%06d.png")],
                       capture_output=True, timeout=120, check=True)
        for path in sorted(frames_dir.glob("*.png")):
            with Image.open(path) as image:
                yield _resize(image)
    except subprocess.CalledProcessError as exc:
        raise ValueError("Video frame extraction failed") from exc
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def _near_duplicate(image: Image.Image, existing: list[tuple[int, str]], threshold: float) -> str | None:
    current = _dhash_image(image)
    for previous, name in existing:
        similarity = 1 - ((current ^ previous).bit_count() / 64)
        if similarity >= threshold:
            return name
    return None


def download(dataset_root: Path, dataset: str, attempt_id: str, keys: list[str], threshold: float) -> dict:
    if not .5 <= threshold <= 1:
        raise ValueError("Near-duplicate threshold must be between 0.50 and 1.00")
    gallery = _attempt_folder(dataset_root, dataset, attempt_id)
    attempt = _manifest(gallery)
    by_key = {item["key"]: item for item in attempt["results"]}
    if not keys or len(keys) > MAX_RESULTS or any(key not in by_key for key in keys):
        raise ValueError("Select one or more results from this scrape attempt")
    accepted = gallery / "filtered"
    before = gallery / "before"
    motion_gallery = gallery / "frames"
    motion_gallery.mkdir(exist_ok=True)
    scrape_root = dataset_root / dataset / ".scrapes"
    previous_galleries = [path / "filtered" for path in scrape_root.iterdir()
                          if path.is_dir() and not path.is_symlink()] if scrape_root.is_dir() else []
    image_folders = [accepted, dataset_root / dataset, *previous_galleries]
    existing_names: set[str] = set()
    existing_hashes: list[tuple[int, str]] = []
    for image_folder in image_folders:
        for existing_path in image_folder.glob("*.png"):
            try:
                if existing_path.is_symlink():
                    continue
                existing_names.add(existing_path.name)
                existing_hashes.append((_dhash(existing_path), existing_path.name))
            except (OSError, ValueError):
                continue
    attempt["threshold"] = threshold
    report = []
    gallery_count = (sum(len(entry.get("names", [])) for entry in attempt.get("before_files", []))
                     + len(attempt.get("frame_files", [])))
    for key in dict.fromkeys(keys):
        item = by_key[key]
        is_motion = item.get("extension", "").lower() in MOTION_EXTENSIONS
        with tempfile.NamedTemporaryFile(prefix=f"grabber-{key}-", suffix=item.get("extension", ".bin"),
                                         delete=False) as temporary:
            raw_path = Path(temporary.name)
        names = []
        frame_names = []
        before_names = []
        try:
            _fetch(item["url"], raw_path)
            for image in _frames(raw_path):
                if gallery_count >= MAX_GALLERY_PNGS:
                    report.append({"key": key, "status": "limit_reached", "limit": MAX_GALLERY_PNGS})
                    break
                gallery_count += 1
                name, png_bytes, _ = _png_name(image)
                if is_motion:
                    destination = motion_gallery / name
                    if not destination.exists():
                        destination.write_bytes(png_bytes)
                        destination.with_suffix(".txt").write_text(
                            ", ".join(item.get("tags", [])), encoding="utf-8")
                    if name not in frame_names:
                        frame_names.append(name)
                    continue
                (before / name).write_bytes(png_bytes)
                if name not in before_names:
                    before_names.append(name)
                if name in existing_names:
                    report.append({"key": key, "status": "duplicate", "name": name})
                    continue
                match = _near_duplicate(image, existing_hashes, threshold)
                if match:
                    report.append({"key": key, "status": "near_duplicate", "name": name,
                                   "match": match, "threshold": threshold})
                    continue
                destination = accepted / name
                destination.write_bytes(png_bytes)
                destination.with_suffix(".txt").write_text(", ".join(item.get("tags", [])), encoding="utf-8")
                names.append(name)
                existing_names.add(name)
                existing_hashes.append((_dhash_image(image), name))
            attempt["files"].extend(names)
            if is_motion:
                attempt.setdefault("frame_files", []).extend(
                    name for name in frame_names if name not in attempt.get("frame_files", []))
                report.append({"key": key, "status": "frames", "files": frame_names})
            else:
                attempt["before_files"].append({"key": key, "names": before_names})
                report.append({"key": key, "status": "filtered", "files": names, "before": before_names})
        except Exception as exc:
            report.append({"key": key, "status": "failed", "message": str(exc)[:300]})
        finally:
            raw_path.unlink(missing_ok=True)
    attempt["status"] = "downloaded"
    (gallery / "attempt.json").write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    return {"results": report, "threshold": threshold}


def gallery(dataset_root: Path, dataset: str, attempt_id: str) -> dict:
    folder = _attempt_folder(dataset_root, dataset, attempt_id)
    attempt = _manifest(folder)
    result_by_key = {item["key"]: item for item in attempt["results"]}
    originals = []
    for entry in attempt.get("before_files", []):
        result = result_by_key.get(entry["key"], {})
        for name in entry.get("names", []):
            if Path(name).name == name and (folder / "before" / name).is_file():
                originals.append({"key": entry["key"], "name": result.get("name"),
                                  "source": result.get("source"), "page_url": result.get("page_url"),
                                  "preview": f"/training/scrape/file/{dataset}/{attempt_id}/before/{name}"})
    filtered = []
    for name in sorted(set(attempt.get("files", []))):
        if Path(name).name != name:
            continue
        path = folder / "filtered" / name
        if path.is_file():
            filtered.append({"name": name, "caption": path.with_suffix(".txt").read_text(encoding="utf-8")
                             if path.with_suffix(".txt").is_file() else "",
                             "preview": f"/training/scrape/file/{dataset}/{attempt_id}/filtered/{name}",
                             "kind": "filtered"})
    frames = []
    for name in sorted(set(attempt.get("frame_files", []))):
        if Path(name).name != name:
            continue
        path = folder / "frames" / name
        if path.is_file():
            frames.append({"name": name, "caption": path.with_suffix(".txt").read_text(encoding="utf-8")
                           if path.with_suffix(".txt").is_file() else "",
                           "preview": f"/training/scrape/file/{dataset}/{attempt_id}/frames/{name}",
                           "kind": "frames"})
    return {"attempt": attempt_id, "query": attempt["query"], "sources": attempt["sources"],
            "threshold": attempt.get("threshold", .95), "before": originals,
            "filtered": filtered, "frames": frames}


def archived_datasets(dataset_root: Path) -> list[dict]:
    found = []
    for folder in sorted(dataset_root.iterdir()):
        if folder.is_symlink() or not folder.is_dir() or folder.name.startswith("."):
            continue
        images = [path for path in folder.iterdir()
                  if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".png"]
        if images:
            found.append({"dataset": folder.name, "images": len(images)})
    return found


def archived_gallery(dataset_root: Path, dataset: str) -> dict:
    candidate = dataset_root / dataset
    if candidate.is_symlink():
        raise ValueError("Dataset directory must not be a symlink")
    folder = candidate.resolve()
    root = dataset_root.resolve()
    if not folder.is_relative_to(root) or folder.is_symlink() or not folder.is_dir():
        raise ValueError("Invalid archived dataset")
    images = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() != ".png" or path.is_symlink() or not path.is_file():
            continue
        caption_path = path.with_suffix(".txt")
        images.append({"name": path.name,
                       "caption": caption_path.read_text(encoding="utf-8") if caption_path.is_file() and not caption_path.is_symlink() else "",
                       "preview": f"/training/dataset/file/{dataset}/{path.name}", "kind": "archived"})
    return {"attempt": f"Archived dataset: {dataset}", "query": "", "sources": ["Training dataset"],
            "threshold": None, "before": [], "filtered": images, "frames": []}


def delete(dataset_root: Path, dataset: str, attempt_id: str, names: list[str],
           frame_names: list[str] | None = None) -> dict:
    folder = _attempt_folder(dataset_root, dataset, attempt_id)
    attempt = _manifest(folder)
    frame_names = frame_names or []
    if (not names and not frame_names) or any(
        Path(name).name != name or name not in attempt.get("files", []) for name in names
    ) or any(Path(name).name != name or name not in attempt.get("frame_files", []) for name in frame_names):
        raise ValueError("Select gallery images or frames to delete")
    for kind, selected in (("filtered", names), ("frames", frame_names)):
        for name in selected:
            for path in (folder / kind / name, (folder / kind / name).with_suffix(".txt")):
                path.unlink(missing_ok=True)
    for name in names:
        attempt["files"].remove(name)
    for name in frame_names:
        attempt["frame_files"].remove(name)
    (folder / "attempt.json").write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    return gallery(dataset_root, dataset, attempt_id)


def save_caption(dataset_root: Path, dataset: str, attempt_id: str, name: str, caption: str,
                 kind: str = "filtered") -> dict:
    folder = _attempt_folder(dataset_root, dataset, attempt_id)
    attempt = _manifest(folder)
    kind_files = {"filtered": "files", "frames": "frame_files"}
    if kind not in kind_files or name not in attempt.get(kind_files[kind], []) or Path(name).name != name:
        raise ValueError("Image is not in this scrape gallery")
    image = folder / kind / name
    if len(caption.strip().encode("utf-8")) > 65536:
        raise ValueError("Caption exceeds 64 KiB")
    image.with_suffix(".txt").write_text(caption.strip(), encoding="utf-8")
    return {"name": name, "caption": caption[:65536].strip()}


def archive(dataset_root: Path, dataset: str, attempt_id: str, names: list[str],
            target_dataset: str | None = None, frame_names: list[str] | None = None) -> dict:
    folder = _attempt_folder(dataset_root, dataset, attempt_id)
    attempt = _manifest(folder)
    frame_names = frame_names or []
    if (not names and not frame_names) or any(
        Path(name).name != name or name not in attempt.get("files", []) for name in names
    ) or any(Path(name).name != name or name not in attempt.get("frame_files", []) for name in frame_names):
        raise ValueError("Select tagged images or frames to submit")
    target_dataset = target_dataset or dataset
    root = dataset_root.resolve()
    destination = (root / target_dataset).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Training dataset path leaves the configured dataset directory")
    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for kind, selected in (("filtered", names), ("frames", frame_names)):
        for name in selected:
            source = folder / kind / name
            caption = source.with_suffix(".txt")
            if not caption.is_file() or not caption.read_text(encoding="utf-8").strip():
                raise ValueError(f"Add a caption before archiving {name}")
            target = destination / name
            target_caption = target.with_suffix(".txt")
            if target.is_symlink() or target_caption.is_symlink():
                raise ValueError(f"Archive target must not be a symlink: {name}")
            if target.exists():
                if hashlib.md5(target.read_bytes(), usedforsecurity=False).hexdigest() != name.split("_", 1)[0]:
                    raise ValueError(f"Archive filename collision: {name}")
                source.unlink(missing_ok=True)
                caption.unlink(missing_ok=True)
            else:
                shutil.move(str(source), str(target))
                shutil.move(str(caption), str(target_caption))
            attempt["files" if kind == "filtered" else "frame_files"].remove(name)
            moved.append(name)
    attempt["status"] = "archived"
    (folder / "attempt.json").write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    return {"archived": moved}
