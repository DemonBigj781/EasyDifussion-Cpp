"""Small real-backend check: requested scale is final, not the model's native scale."""
import base64
import io
import json
import os
from urllib.request import Request, urlopen
from PIL import Image

source = Image.new("RGB", (16, 16), (30, 100, 180))
buffer = io.BytesIO()
source.save(buffer, format="PNG")
payload = {"imageList": [{"name": "scale-regression", "data": base64.b64encode(buffer.getvalue()).decode()}],
           "upscaling_resize": 2, "upscaler_1": os.environ.get("TEST_UPSCALER", "RealESRGAN_x4plus_anime_6B")}
request = Request(os.environ.get("TEST_BACKEND", "http://localhost:7860/v1") + "/sdapi/v1/extra-batch-images",
                  json.dumps(payload).encode(), {"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    result = json.load(response)
images = result["images"]
image = images[0]
if isinstance(image, dict):
    image = image["image"]
image = image.split(",", 1)[-1]
with Image.open(io.BytesIO(base64.b64decode(image))) as output:
    assert output.size == (32, 32), f"Requested 2x of 16x16, got {output.size} using {payload['upscaler_1']}"
print("PASS: requested 2x produces 32x32 from 16x16 using", payload["upscaler_1"])
