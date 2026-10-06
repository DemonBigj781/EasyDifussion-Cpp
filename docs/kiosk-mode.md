# Kiosk mode

Open **Settings → Kiosk mode**, select **Enable kiosk mode**, then choose
**Save kiosk mode**. The separate save button changes only this policy; it does
not reset backend, model-directory, or VRAM settings. Enabling requires an idle
generation backend and no active training or Perchance operation.

The setting is persisted as the boolean `kiosk_mode` in `config.yaml`. It defaults
to off. Turning it off restores normal model catalogs, saved LoRA choices and
the previously configured local gallery directory. Neither model files nor local
gallery files are deleted or replaced.

## Restrictions

The model picker and rendering API allow only these installed checkpoint IDs:

| Family | Checkpoint ID |
| --- | --- |
| SD1.5 | `1.5/sd-v1-5` |
| SDXL | `sdxl/sd_xl_base_1.0` |
| Anima | `Anima/anima-base-v1.0` |

This is an exact allowlist, not an architecture-tag filter: fine-tunes in the
same families are excluded. An absent approved checkpoint is not replaced with
a different model. Catalog filtering does not change underlying model metadata.

LoRA controls are hidden, saved LoRAs are not loaded into the kiosk UI, the LoRA
catalog is empty, and generation requests with LoRA selections or inline
`<lora:...>` / `<lyco:...>` syntax are rejected. These checks are server-side,
not just hidden buttons. Training, model download/merge, model-file browsing,
dataset gallery, tagging, video and legacy UI routes are unavailable in kiosk
mode because they expose unrestricted model or image surfaces.

## Perchance

Perchance provides `g`, `pg13`, and `none` filters, not a PG filter. Kiosk mode
forces the stricter **G** level, regardless of a client requesting `none`.

The existing browser-backed Perchance client verifies that the public gallery
frame and filter selector are set to G before reading its feed. Individual image
lookups also validate the provider's `nsfw`, `shocking` and `pg13Soft` metadata
inside that browser context. Kiosk requests use this supported filtered client
path, not blocked raw-document HTTP requests.

The server accepts previews only from that G-filtered result, with matching
image identity, channel, public-feed marker and supported image origin. Explicit
unsafe flags or invalid/missing provenance cause suppression. List responses
include `suppressed_count`. Images from unfiltered requests and previously
cached, unverified images cannot be fetched directly through the kiosk cache
route. Verified cache permissions expire after ten minutes and are cleared when
the mode changes.

Perchance generation does not supply a verified content rating. Therefore its
image-generation page/API, recent unrated outputs, local output file routes and
save-to-gallery operation are unavailable while kiosk mode is on. The
**Perchance Gallery** remains available with the enforced G policy. Turning
kiosk mode off restores the previous generation and gallery behavior.

These are provider metadata checks, not an independent visual classifier or a
guarantee that every provider label is accurate.

## Main image gallery

Kiosk mode displays film-frame images from **Destockd**, using its public random
feed with the website's default exclusions. It does not request the option to
include disturbing/blank or excluded footage. **Refresh** requests another batch
of up to 48 frames. Cards link back to their source on Destockd.

The local directory is not scanned. Directory editing, local file/thumbnail
routes, deletion, selection and collage controls are unavailable in kiosk mode.
If Destockd fails, the gallery shows an error and clears stale cards; it never
falls back to local images. Local directory preferences are retained for normal
mode.

Destockd's source filtering is **not a PG certification**. It is an archive of
historical footage and may contain material unsuitable for some audiences.

## Policy propagation and limits

C++ pages read the current server policy before initializing model or gallery
controls. Saving the setting reloads other tabs on the same origin; pages also
check every five seconds for changes from other browsers/devices. Navigation
from the browser back/forward cache reloads the page. An unavailable policy
leaves the content hidden with an explanatory error, rather than loading local
images or unrestricted model lists.

This is a **display and generation policy**, not a password-protected account
boundary, full-screen browser lockdown, or operating-system kiosk. Anyone with
access to its settings endpoint can turn it off. It cannot revoke images already
downloaded before activation, and it does not certify locally generated base-model
outputs as PG. A managed public terminal needs separate access controls.

## API and verification

- `GET /kiosk`: current boolean policy, approved checkpoint IDs and source/filter
  information.
- `POST /kiosk` with `{"enabled": true}` or `{"enabled": false}`: persist the
  policy. Other value types are rejected. Busy activation returns HTTP 409.
- General `/app_config` updates cannot set `kiosk_mode`; the dedicated endpoint
  provides validation and the idle guard.
- Existing model, render, gallery and Perchance endpoints enforce the policy.

Run the policy/ASGI tests in the project's Python environment:

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/test_kiosk_mode.py`

Existing Perchance behavior has a separate regression suite:

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/test_perchance_image_batch.py`

`scripts/test_cpp_kiosk_ui.cjs` uses Playwright to cover both directions of mode
switching, preserved preferences, model/LoRA payload restrictions, remote gallery
failure without local fallback, rating mismatches, policy failures and mobile
layout. Its policy/data requests are fixtures, and render submissions are
intercepted: it does not change real settings or start generation.
