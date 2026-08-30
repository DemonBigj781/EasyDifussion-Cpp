#!/usr/bin/env python3
"""
Simple Python script to call the Stable Diffusion API and generate images.
Scripted test version that sets a model and generates an image.
"""

import requests
import json
import base64
import os
from datetime import datetime


def set_model(model_name, port=8188):
    """
    Set the Stable Diffusion model via the API.

    Args:
        model_name (str): Name of the model checkpoint
        port (int): API server port (default: 8188)

    Returns:
        bool: True if successful, False otherwise
    """
    url = f"http://localhost:{port}/v1/sdapi/v1/options"

    payload = {"sd_model_checkpoint": model_name}

    print(f"Setting model to: {model_name}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print("Model set successfully")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to set model: {e}")
        return False


def generate_image(
    prompt, negative_prompt="", width=512, height=512, steps=20, cfg_scale=7.0, seed=-1, batch_size=1, port=8188
):
    """
    Generate an image using the Stable Diffusion API.

    Args:
        prompt (str): The text prompt for image generation
        negative_prompt (str): The negative prompt (optional)
        width (int): Image width (default: 512)
        height (int): Image height (default: 512)
        steps (int): Number of inference steps (default: 20)
        cfg_scale (float): Classifier-free guidance scale (default: 7.0)
        seed (int): Random seed (-1 for random, default: -1)
        batch_size (int): Number of images to generate (default: 1)
        port (int): API server port (default: 8188)

    Returns:
        list: List of generated image file paths
    """
    url = f"http://localhost:{port}/v1/sdapi/v1/txt2img"

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "batch_size": batch_size,
    }

    print(f"Sending request to {url}")
    print(f"Prompt: {prompt}")
    print(f"Parameters: {width}x{height}, steps={steps}, cfg_scale={cfg_scale}, seed={seed}")

    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        response.raise_for_status()

        result = response.json()

        if "images" not in result:
            print("Error: No images in response")
            print(f"Response: {result}")
            return []

        # Create output directory if it doesn't exist
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, image_base64 in enumerate(result["images"]):
            # Decode base64 image
            image_data = base64.b64decode(image_base64)

            # Save image
            filename = f"{output_dir}/image_{timestamp}_{i+1}.png"
            with open(filename, "wb") as f:
                f.write(image_data)

            saved_files.append(filename)
            print(f"Saved image: {filename}")

        # Print generation info
        if "info" in result:
            info = json.loads(result["info"])
            print(f"Generation info: seed={info.get('seed', 'unknown')}")

        return saved_files

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def main():
    # Check if requests library is available
    try:
        import requests
    except ImportError:
        print("Error: requests library is required. Install with: pip install requests")
        return

    # Set the model (change this to your actual model name)
    model_name = "sd-v1-5.safetensors"  # Replace with your model filename
    if not set_model(model_name):
        print("Failed to set model, aborting test")
        return

    # Generate test image
    files = generate_image(
        prompt="a beautiful landscape with mountains and a lake",
        negative_prompt="",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.0,
        seed=42,
        batch_size=1,
        port=8188,
    )

    if files:
        print(f"\nSuccessfully generated {len(files)} image(s)")
        for file in files:
            print(f"  {file}")
    else:
        print("\nFailed to generate images")


if __name__ == "__main__":
    main()
