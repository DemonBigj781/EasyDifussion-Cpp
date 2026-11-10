# Online model browser and image metadata

Use **Models** to see each matching Civitai model together with its bound images.
Each result has an expandable **Model files** section for downloads and an
automatically loaded image gallery.
The initial gallery contains up to six images. **Previous images** and
**Next images** page only that model's gallery; the bottom **Prev/Next** buttons
page the model results independently.

Automatic galleries load sequentially to avoid a burst of provider requests.
An image-listing failure stays in its own card with **Retry images**, without
hiding model downloads or other galleries. New searches and provider changes
cancel the old gallery requests. Galleries span model versions unless an exact
version was supplied in the search URN. Image sort, post ID, username, and
selected image tags are image-only filters. The image API requests both
generation metadata and tag details and preserves them in its report.
Changes to filters take effect when **Search** is pressed,
not halfway through paging existing results. Hugging Face remains a model-file
browser and does not query Civitai for images.

Use **Images** for image-only browsing through `/api/v1/images`, not the old
model-location search. Model cards and downloads are hidden in this view.
The bottom **Prev/Next** buttons page the image feed. **Random** is available
only here; provider restrictions or errors are shown rather than silently
changing the requested sort.

## Selection-only tag autocomplete

Type in **Select image tags** and choose a suggestion with the mouse or
Arrow keys/Enter. Selected tags appear as removable chips. Only their numeric
IDs are submitted; unselected text blocks Search and is never used as a query.
The free-text model search is hidden in Images, while Models still supports
model names and exact model/version AIR identifiers.

Suggestions come from verified `{id, name}` pairs in image responses using
`withTags=true`, not the model-tag names returned by `/api/v1/tags` (which omit
image tag IDs). A small image-feed request seeds the catalog on load, then
browsing discovers more tags. The catalog and selected chips are cached across
loads. It is a discovered catalog, not a complete global image-tag index:
unseen tags cannot be selected until they appear in fetched image metadata.
The app never invents numeric IDs from tag names.

Model enums are cached in browser storage. Each load restores those options
immediately, then refreshes the enum endpoint and saves the successful response.
If the provider is unavailable, cached options and saved selections remain
usable alongside built-in defaults. The cache includes model types, model-file
types, base models, active base models, and base-model types; image tag IDs are
not model enums.

## Two independent metadata sources

Choose **Prompt & metadata** on an image. The dialog has separate, selectable,
copyable report fields:

- **Civitai-reported metadata** contains the image record returned by Civitai,
  including its generation metadata. It is not evidence of what is embedded
  inside the delivered image.
- **Downloaded-file metadata** is populated by **Inspect downloaded file**.
  This fetches the displayed image URL and inspects those exact bytes without
  re-encoding or saving them to the gallery. The report includes byte size,
  SHA-256, detected format, dimensions, PNG text fields (including workflow
  JSON), EXIF and UserComment, and available XMP in the image info. Binary
  metadata values are retained as base64 with decoded text where appropriate.

Civitai can deliver a resized or stripped image. The downloaded-file report
describes the delivered file, not a guaranteed copy of the author's original
upload. Empty metadata sections are not filled from the Civitai report.
Re-reading a URL can produce a different file; use the SHA-256 to identify it.

Each report has **Copy report**, **Copy prompt**, **Copy negative**, and
**Apply settings** controls. Applying one report never merges in fields from
the other. Recognized generation fields are applied to the Image settings;
unavailable sampler/scheduler choices stay unchanged and are reported. Common
Automatic1111 parameter text, Easy Diffusion JSON, and direct prompt/settings
fields can be read from embedded metadata. ComfyUI graphs and unknown fields
remain in the report for copying; they are not guessed into a generation task.
Applying settings does not start generation.

**Save + apply Civitai settings** explicitly imports the website's settings.
It saves the fetched image unchanged, plus a `.civitai.json` sidecar containing
both the Civitai record and metadata extracted from that same saved file.

Metadata inspection and image import accept only official Civitai HTTPS hosts,
validate every redirect before following it, and enforce a 100 MB file limit.
GPS or other personal metadata embedded in a file will appear in its report;
review it before copying or sharing.

## Diagnostics

The server logs status, received byte count, declared `Content-Length`, content
encoding, elapsed request time, and parsed result count for provider API calls.
URLs in these diagnostics omit query strings; prompts, authentication headers,
tokens, and complete response bodies are not logged. `requests` decompresses
API response content, so received bytes can legitimately differ from a
compressed `Content-Length`. Image/model transfer logs report received bytes;
model logs also distinguish resumed bytes from total file size.
