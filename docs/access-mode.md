# Website and REST access modes

Secure mode is the default. The site shows a sign-in page until you enter the
server's bearer key. REST clients send `Authorization: Bearer <key>` on every
request. No key is built in; secure mode does not expose the UI, REST, or static
files if the server has no configured key.

The browser sends the key once to sign in. Then it sends a short-lived,
same-origin HttpOnly session credential with its requests. This keeps the raw
key out of browser storage and URLs. Sessions expire after eight hours; replacing
the configured key invalidates earlier sessions. Saved-history API clients still
need the history bearer key and its separate explicit enablement. Use the same
key file for both if desired.

## Set up secure mode on a local POSIX machine

Before starting Easy Diffusion:

```sh
umask 077
install -d -m 700 "$HOME/.config/easy-diffusion-keys"
export EASY_DIFFUSION_BEARER_TOKEN_FILE="$HOME/.config/easy-diffusion-keys/server"
# Provision once. A pre-existing key is never overwritten by this command.
(set -C; openssl rand -hex 32 > "$EASY_DIFFUSION_BEARER_TOKEN_FILE")
./start.sh
```

On platforms without POSIX file ownership checks, provision the same 64-character
hex secret in the server process's `EASY_DIFFUSION_BEARER_TOKEN` environment
variable without printing it or adding it to YAML. Don't set a token-file
variable when using the environment value. An existing
`EASY_DIFFUSION_RUN_HISTORY_TOKEN_FILE` is also accepted if no primary token file
is set.

Keep the service on localhost unless you deliberately set up remote access.
Secure mode accepts direct loopback HTTP or HTTPS. For HTTPS termination, use a
trusted proxy that preserves the request Host and supplies the HTTPS scheme.
Never expose the separate native sdkit port: it remains a loopback backend, not
the authenticated Legacy HTTP server.

## Developer mode

For an attended local debugging session only:

```sh
EASY_DIFFUSION_DEV_MODE=1 ./start.sh
```

This enables detailed console diagnostics and the redacted debug file. It allows
unauthenticated direct loopback UI and regular REST requests, rejects cross-origin
browser requests and public tunnels, and the standard launcher binds it to
loopback even if saved settings request a network bind. Never forward that
development port. Private saved history still has its own key and enable flag.
Restart without the environment toggle to return to secure mode. Existing debug
files are not deleted; handle them as private.
