import requests
from PIL import Image
import io
import base64
import os
import sys
import random
import time

if "SD2_DEPTH_API_URL" not in os.environ:
    print("SD2_DEPTH_API_URL env variable must be set")
    sys.exit(1)

prompt = "glossy white kitchen, fridge, old-style chairs, painting on the wall, concrete trowel, modern table with modern white chairs, mate glass entrance door, teak floor, air-conditioning"
negative_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"
steps = 20
seed = random.randint(10000, 99999)

def prepare_image_data(path):
    with open(path, "rb") as f:
        im_bytes = f.read()

    return base64.b64encode(im_bytes).decode("utf8")

print("Processing started")
start = time.time()

res = requests.post(os.environ["SD2_DEPTH_API_URL"], json={
    "base_image": prepare_image_data('test.jpg'),
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "strength": 1.0,
    "steps": steps,
    "seed": seed
})

end = time.time()
print(f"Processing finished in: {round(end - start, 2)}s")

if res.ok:
    image = Image.open(io.BytesIO(res.content))
    random_ident = random.randint(10000, 99999)
    image.save(f"res_{random_ident}.png")
