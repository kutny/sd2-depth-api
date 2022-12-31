import requests
from PIL import Image
import io
import base64
  
prompt = "glossy white kitchen, fridge, old-style chairs, painting on the wall, concrete trowel, modern table with modern white chairs, mate glass entrance door, teak floor, air-conditioning"
negative_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"
steps = 20
seed = 128
url = "http://3e51-34-87-119-52.ngrok.io/"

def prepare_image_data(path):
    with open(path, "rb") as f:
        im_bytes = f.read()        
    
    return base64.b64encode(im_bytes).decode("utf8")

res = requests.post(url, json={
    "base_image": prepare_image_data('test.jpg'),
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "strength": 1.0,
    "steps": steps,
    "seed": seed
})

if res.ok:
    image = Image.open(io.BytesIO(res.content))
    image.save("res.png")
