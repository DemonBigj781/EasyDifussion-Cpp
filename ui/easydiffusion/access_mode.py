"""Secure-by-default gate for the Legacy website and its REST API."""
import hashlib
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import stat
import time

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

TOKEN = re.compile(rb"[0-9A-Fa-f]{64}")
SESSION = re.compile(r"[0-9]{1,12}\.[0-9a-f]{32}\.[0-9a-f]{64}")
SESSION_SECONDS = 8 * 60 * 60
NO_STORE = {"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff"}
AUTH_REQUIRED = {**NO_STORE, "WWW-Authenticate": "Bearer"}


def dev_mode():
    return os.environ.get("EASY_DIFFUSION_DEV_MODE") == "1"


def _key():
    filename = (os.environ.get("EASY_DIFFUSION_BEARER_TOKEN_FILE")
                or os.environ.get("EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE"))
    if filename:
        path = Path(filename)
        try:
            info = path.lstat()
            if (not path.is_absolute() or not stat.S_ISREG(info.st_mode)
                    or getattr(os, "getuid", lambda: None)() != info.st_uid
                    or info.st_mode & 0o077):
                return None
            with path.open("rb") as stream:
                raw = stream.read(68).rstrip(b"\r\n")
        except OSError:
            return None
    else:
        raw = os.environ.get("EASY_DIFFUSION_BEARER_TOKEN", "").encode("utf-8")
    return raw if TOKEN.fullmatch(raw) else None


def _loopback(host):
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _local(request):
    # Require the actual ASGI peer and the address in the request to be local.
    # Public proxy/Host requests must not enter the unauthenticated developer mode.
    return (request.client is not None and _loopback(request.client.host)
            and _loopback(request.url.hostname)
            and not any(header in request.headers for header in
                        ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "cf-connecting-ip")))


def _bearer(request, key):
    scheme, _, value = request.headers.get("Authorization", "").partition(" ")
    supplied = value.encode("utf-8")
    return (scheme.lower() == "bearer" and TOKEN.fullmatch(supplied) is not None
            and hmac.compare_digest(key, supplied))


def _sign(payload, key):
    return hmac.new(key, b"easy-diffusion-browser-session-v1\0" + payload.encode("ascii"), hashlib.sha256).hexdigest()


def _session_cookie(request):
    return "__Host-ed-session" if request.url.scheme == "https" else "ed-local-session"


def _read_session(request, key):
    value = request.cookies.get(_session_cookie(request), "")
    if not SESSION.fullmatch(value):
        return False
    expiry, nonce, signature = value.split(".")
    now = int(time.time())
    if not now < int(expiry) <= now + SESSION_SECONDS:
        return False
    return hmac.compare_digest(signature, _sign(f"{expiry}.{nonce}", key))


def _same_origin(request):
    # Login and cookie-authenticated mutations must not be initiated by another site.
    origin = request.headers.get("Origin")
    if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
        return False
    if origin is not None:
        return origin == f"{request.url.scheme}://{request.url.netloc}"
    return request.method in ("GET", "HEAD", "OPTIONS")


def _login_page(status):
    nonce = secrets.token_urlsafe(24)
    heading = "Sign in to Easy Diffusion" if status == 401 else "Set up Easy Diffusion access"
    body = (
        '<form id="login"><label>Bearer key <input id="key" type="password" '
        'autocomplete="off" required autofocus></label> <button>Sign in</button></form>'
        '<p id="error" role="alert"></p>'
        f'<script nonce="{nonce}">' + """
const form = document.getElementById("login");
const key = document.getElementById("key");
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const bearer = key.value.trim();
  key.value = "";
  try {
    const response = await fetch("/auth/login", {
      method: "POST", headers: {Authorization: "Bearer " + bearer},
      credentials: "same-origin", cache: "no-store"
    });
    if (response.ok) { location.replace("/"); return; }
  } catch (_) {}
  document.getElementById("error").textContent = "Sign-in failed. Check the key and connection.";
});
""" + "</script>"
        if status == 401 else "<p>Ask the server operator to configure a bearer key, then reload.</p>"
    )
    return HTMLResponse(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>{heading}</title><h1>{heading}</h1>{body}</html>',
        status_code=status, headers={**NO_STORE,
            "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; connect-src 'self'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'"},
    )


def _reject(status, detail, authenticated=False):
    return JSONResponse({"detail": detail}, status_code=status,
                        headers=AUTH_REQUIRED if authenticated else NO_STORE)


def install(app):
    @app.middleware("http")
    async def access_gate(request: Request, call_next):
        path = request.url.path
        browser_entry = request.method == "GET" and path in ("/", "/legacy")
        local = _local(request)
        if dev_mode():
            if not local:
                return _reject(403, "Developer mode only accepts direct loopback requests.")
            if (request.headers.get("Origin") is not None or request.headers.get("Sec-Fetch-Site") is not None) and not _same_origin(request):
                return _reject(403, "Developer mode does not accept cross-origin browser requests.")
            return await call_next(request)

        if request.url.scheme != "https" and not local:
            return _reject(403, "Use HTTPS or a direct loopback connection.")
        key = _key()
        if key is None:
            if browser_entry:
                return _login_page(503)
            return _reject(503, "Server bearer key is not configured.")
        bearer = _bearer(request, key)

        if path == "/auth/login" and request.method == "POST":
            if not bearer:
                return _reject(401, "Bearer key required.", authenticated=True)
            if not _same_origin(request):
                return _reject(403, "Login requires the website origin.")
            expires = int(time.time()) + SESSION_SECONDS
            payload = f"{expires}.{secrets.token_hex(16)}"
            response = JSONResponse({"signed_in": True}, headers=NO_STORE)
            response.set_cookie(_session_cookie(request), payload + "." + _sign(payload, key),
                                httponly=True, secure=request.url.scheme == "https",
                                samesite="strict", max_age=SESSION_SECONDS, path="/")
            return response

        same_site_browser = (
            request.headers.get("Sec-Fetch-Site") == "same-origin"
            or request.headers.get("Origin") == f"{request.url.scheme}://{request.url.netloc}"
            or browser_entry
        )
        session = same_site_browser and _same_origin(request) and _read_session(request, key)
        if path == "/auth/logout" and request.method == "POST":
            if not session and not bearer:
                return _reject(401, "Authentication required.", authenticated=True)
            response = JSONResponse({"signed_out": True}, headers=NO_STORE)
            response.delete_cookie(_session_cookie(request), path="/", secure=request.url.scheme == "https",
                                   httponly=True, samesite="strict")
            return response
        if not bearer and not session:
            if browser_entry:
                return _login_page(401)
            return _reject(401, "Bearer key required.", authenticated=True)
        response = await call_next(request)
        response.headers.update(NO_STORE)
        return response
