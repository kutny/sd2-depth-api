import base64
import torch
import uuid
from flask import Flask, request, Response, send_file
import io
import os
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, EulerAncestralDiscreteScheduler
import random
import colorlog
import sys
from logging import PercentStyle

class ExtraFieldsFormatter(colorlog.ColoredFormatter):
    def __init__(self, *args, **kwargs):
        self.__orig_fmt = args[0]

        super().__init__(*args, **kwargs)

    def format(self, record):
        extra_keys = ExtraKeysResolver.get_extra_keys(record)

        if not extra_keys:
            return super().format(record)

        def map_placeholder(field_name):
            return "{}: %({})s".format(field_name, field_name)

        extra_keys_placeholders = list(map(map_placeholder, extra_keys))

        self.__set_format(self.__orig_fmt + "\n" + "{" + ", ".join(extra_keys_placeholders) + "}")
        formated = super().format(record)
        self.__set_format(self.__orig_fmt)

        return formated

    def __set_format(self, fmt: str):
        self._fmt = fmt
        self._style = PercentStyle(self._fmt)

class ExtraKeysResolver:

    ignored_record_keys = [
        "name",
        "msg",
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    ]

    @staticmethod
    def get_extra_keys(record):
        return record.__dict__.keys() - ExtraKeysResolver.ignored_record_keys

def create_logger(name):
    handler = colorlog.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter('%(log_color)s%(message)s'))

    logger = colorlog.getLogger(name)
    logger.setLevel(20) # info
    logger.addHandler(handler)

    return logger

logger = create_logger("sd2-depth-api")

if len(sys.argv) != 2:
    logger.error("Missing model path argument")
    sys.exit(1)

model_path = sys.argv[1]

base_images_dir = f"{os.getcwd()}/input_images"

os.mkdir(base_images_dir)

app = Flask(__name__)

logger.info(f"Loading model {model_path}")

scheduler = EulerAncestralDiscreteScheduler.from_config(
    "stabilityai/stable-diffusion-2-depth",
    subfolder="scheduler",
)

depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    scheduler=scheduler,
).to("cuda")

def load_image(im_b64) -> Image:
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    return Image.open(io.BytesIO(img_bytes))

logger.info("sd2-depth-api started")

@app.route('/', methods=['POST'])
def generate_image():
    request_id = str(uuid.uuid4())
    params = request.json

    if params is None:
        params = dict()

    params_logging = params.copy()
    del params_logging["base_image"]

    logger.info("New request", extra={**{"request_id": request_id}, **params_logging})

    if "base_image" not in params:
        return Response("base_image must be provided", status=400)

    base_image = load_image(params['base_image'])

    base_image_save_path = f"{base_images_dir}/{request_id}.png"
    base_image.save(base_image_save_path)

    if "prompt" not in params:
        return Response("prompt must be provided", status=400)

    prompt = params['prompt']

    seed = params['seed'] if "steps" in params else random.randint(1000, 9999)
    negative_prompt = params['negative_prompt'] if "negative_prompt" in params else None
    guidance_scale = params['guidance_scale'] if "guidance_scale" in params else 7
    strength = params['strength'] if "strength" in params else 0.75
    steps = params['steps'] if "steps" in params else 20
    
    logger.info(f'Params processed', extra={
        "request_id": request_id,
        "base_image_path": base_image_save_path,
        "prompt": prompt,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "steps": steps,
    })

    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    with torch.autocast("cuda"), torch.inference_mode():
        if depth2img_pipe is None:
            raise Exception("Model not loaded, have you set the load_model global variable?")

        image = depth2img_pipe(
            prompt=prompt,
            image=base_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]

    image_io = io.BytesIO()
    image.save(image_io, 'PNG')
    image_io.seek(0)

    return send_file(image_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
