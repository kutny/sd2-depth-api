import base64
import io
from PIL import Image

def load_image(im_b64) -> Image:
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    return Image.open(io.BytesIO(img_bytes))
