# Private run history (opt-in)

The Legacy application API can retain selected video results without regenerating them.
Nothing is archived by default. This does not configure the separate native sdkit API.
Ordinary render responses are unchanged. The updated native engine no longer logs prompt text
or token strings in its generation and tokenization messages. Older logs are unchanged. An artifact can
itself reveal generated content or contain embedded metadata, so treat its bytes as private.

## Local setup

Run history currently requires POSIX file ownership and permissions. Use a private directory outside all web/static, model, plugin, gallery and synced paths.
Create the secret before enabling the feature. Run this as the application user:

```sh
umask 077
install -d -m 700 "$HOME/.local/state/easy-diffusion-history"
install -d -m 700 "$HOME/.config/easy-diffusion-history"
export EASY_DIFFUSION_RUN_HISTORY_DIR="$HOME/.local/state/easy-diffusion-history"
export EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE="$HOME/.config/easy-diffusion-history/token"
# Only create a token on first setup; do not print or paste it into chat.
(set -C; openssl rand -hex 32 > "$EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE")
export EASY_DIFFUSION_RUN_HISTORY=1
```

Inject these settings into the Legacy app process at launch. Keep the app on localhost.
If a remote client is needed, use a private HTTPS endpoint or an SSH tunnel. Keep the
token out of URLs, shell history, proxy logs, and browser scripts. Use a dedicated token.
Unset the flag or remove the token file to disable access. The app rereads the token
file each request; you can rotate it with a private replacement file and atomic rename.
Both the archive directory and token must belong to the application user and must not
be accessible to other users. Do not put the token inside the archive. You may set
`EASY_DIFFUSION_RUN_HISTORY_BUILD` to a verified engine build ID; it is omitted otherwise.

Send `Authorization: Bearer <token>` on every request:

- `GET /v1/runs?limit=20&offset=0`: newest runs, technical settings only.
- `GET /v1/runs/{id}`: technical details and protected artifact URLs.
- `GET /v1/runs/{id}/artifacts/{name}`: a private binary download.

To retain a Legacy `POST /video` result, send the same bearer header and
`"retain_history": true` in the request. Ordinary and temporary-output requests
do not write history. The finished response includes a `run_id` on success, or
`history_saved: false` if the archive write failed without failing the render.
An individual archive is limited to 128 MiB and 513 files. There is no automatic
retention policy; remove the appropriate private run directory locally when no
longer needed. Collection and retrieval through Legacy are unavailable in kiosk mode.

To import a specific existing render as the local owner (without rerendering):

```sh
python scripts/import_run_history.py --archive "$EASY_DIFFUSION_RUN_HISTORY_DIR" \
  --settings /path/to/verified-settings.json /path/to/selected-render.gif
```

The importer does not discover runs or infer settings from filenames. Metadata is
restricted to a technical allowlist. Omit anything unknown. It copies media bytes as-is.
