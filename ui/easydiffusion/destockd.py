"""Read Destockd's public, default-filtered film-frame catalog without local fallback."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fastapi import HTTPException

ORIGIN = "https://destockd.com"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def fetch_catalog(count):
    # Never request include_homepage_excluded or the admin-only excluded feed.
    request = Request(ORIGIN + "/api/random?" + urlencode({"count": count}),
                      headers={"Accept": "application/json", "User-Agent": "EasyDiffusion-Kiosk/1.0"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=12) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError("catalog too large")
            return json.loads(payload)
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Destockd is unavailable. Local images will not be shown in kiosk mode.") from error


def list_images(page=1, page_size=60):
    if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 120:
        raise HTTPException(status_code=400, detail="Invalid gallery page or page_size.")
    payload = fetch_catalog(min(page_size, 48))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise HTTPException(status_code=502, detail="Invalid Destockd catalog; local fallback is disabled.")
    images = []
    for item in payload["results"][:48]:
        if not isinstance(item, dict):
            raise HTTPException(status_code=502, detail="Invalid Destockd image record.")
        if item.get("excluded") or item.get("homepage_excluded"):
            continue
        film, shot, keyframe = (item.get(key) for key in ("film", "shot", "keyframe"))
        if not all(isinstance(value, str) and value for value in (film, shot, keyframe)):
            raise HTTPException(status_code=502, detail="Invalid Destockd image record.")
        parsed = urlsplit(keyframe)
        decoded = unquote(parsed.path)
        if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or not decoded.startswith("/keyframes/") or "\\" in decoded
                or any(part in {"..", "."} for part in decoded.split("/"))
                or not decoded.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
            raise HTTPException(status_code=502, detail="Destockd returned an unsupported image URL.")
        url = ORIGIN + keyframe
        images.append({"id": f"destockd:{film}/{shot}", "filename": f"{film} — {shot}",
                       "url": url, "thumbnail_url": url,
                       "source_url": ORIGIN + "/#/shot/" + quote(film, safe="") + "/" + quote(shot, safe="")})
    return {"source": "destockd", "directory": ORIGIN, "exists": True, "images": images,
            "total": len(images), "page": 1, "page_size": min(page_size, 48), "total_pages": 1,
            "has_previous": False, "has_next": False, "start_index": 1 if images else 0,
            "end_index": len(images), "notice": "Destockd film frames. Refresh for another selection; source filtering is not a PG certification."}
