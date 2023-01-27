import base64
import io
from PIL import Image

def decode_image(im_b64):
    return io.BytesIO(base64.b64decode(im_b64.encode('utf-8')))

def load_image(im_b64) -> Image:
    img_bytes = decode_image(im_b64)
    return Image.open(img_bytes)
