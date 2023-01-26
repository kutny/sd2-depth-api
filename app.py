import base64
import torch
import uuid
from flask import Flask, request, Response, send_file, make_response
import io
import os
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, EulerAncestralDiscreteScheduler
import random
import colorlog
import sys
import logging
from logzio.handler import LogzioHandler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import boto3

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
        self._style = logging.PercentStyle(self._fmt)

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

def download_depth_map(depth_map_url: str, depth_map_path: str):
    # example: https://s3.eu-west-1.amazonaws.com/interiorgen.dev/depth/b702ce1aa3dbe99ffa0e267d468b114e.npy
    url_parts = depth_map_url.split("/")
    key = "/".join(url_parts[-2:])
    bucket_name = url_parts[3:][0]

    s3.download_file(Bucket=bucket_name, Key=key, Filename=depth_map_path)

def get_depth_map(depth_map_path: str):
    depth_map = np.load(depth_map_path)
    depth_map_normalized = np.log(depth_map)

    return depth_map, depth_map_normalized

def to_depthmap_tensor(init_depth: np.ndarray):
    init_depth = np.expand_dims(init_depth, axis=0)
    init_depth = torch.from_numpy(init_depth)
    init_depth = 2. * init_depth - 1.

    return init_depth

def get_depth_map_img(depth_map: np.ndarray):
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    max_val = 255

    out = (depth_map - depth_min) / (depth_max - depth_min)
    # print("depth interval <" + str(out.min()) + "; " + str(out.max()) + ">") # <0; 1>
    out = max_val * out
    # print("depth interval <" + str(out.min()) + "; " + str(out.max()) + ">") # <0; 255>

    img = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
    return Image.fromarray(img)

def save_histogram(depth_map: np.ndarray, title: str, path: str):
    a = np.concatenate(depth_map, axis=0)
    plt.hist(a, bins="auto")
    plt.title(title) 
    plt.savefig(path)

def create_logger(name):
    handler = colorlog.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter('%(log_color)s%(message)s'))

    logger = colorlog.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    app_env = os.environ["APP_ENV"]

    logz_formatter = logging.Formatter('{"app_env": "' + app_env + '"}', validate=False)

    logz_handler = LogzioHandler(os.environ["LOGZIO_TOKEN"], url="https://listener-eu.logz.io:8071")
    logz_handler.setLevel(logging.INFO)
    logz_handler.setFormatter(logz_formatter)
    logger.addHandler(logz_handler)

    return logger

logger = create_logger("sd2-depth-api")

if len(sys.argv) != 2:
    logger.error("Missing model path argument")
    sys.exit(1)

model_path = sys.argv[1]

inputs_dir = f"{os.getcwd()}/inputs"

if not os.path.exists(inputs_dir):
    os.mkdir(inputs_dir)

s3 = boto3.client("s3")

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

    request_dir = f"{inputs_dir}/{request_id}"
    os.mkdir(request_dir)

    base_image_save_path = f"{request_dir}/1-base_image.png"
    base_image.save(base_image_save_path)

    if "prompt" not in params:
        return Response("prompt must be provided", status=400)

    prompt = params['prompt']

    depth_map_url = params['depth_map_url']
    seed = params['seed'] if "steps" in params else random.randint(1000, 9999)
    negative_prompt = params['negative_prompt'] if "negative_prompt" in params else None
    guidance_scale = params['guidance_scale'] if "guidance_scale" in params else 7
    strength = params['strength'] if "strength" in params else 0.75
    steps = params['steps'] if "steps" in params else 20
    order = params['order'] if "order" in params else None

    logger.info(f'Params processed', extra={
        "request_id": request_id,
        "base_image_path": base_image_save_path,
        "depth_map_url": depth_map_url,
        "prompt": prompt,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "steps": steps,
        "order": order
    })

    depth_map_path = f"{request_dir}/2-depth_map.npy"
    depth_map_image_path = f"{request_dir}/3-depth_map.png"
    depth_map_histogram_path = f"{request_dir}/4-depth_map_histogram.png"
    depth_map_normalized_image_path = f"{request_dir}/5-normalized_depth_map.png"
    depth_map_normalized_histogram_path = f"{request_dir}/6-normalized_depth_map_histogram.png"

    download_depth_map(depth_map_url, depth_map_path)
    depth_map, depth_map_normalized = get_depth_map(depth_map_path)
    
    get_depth_map_img(depth_map).save(depth_map_image_path)
    save_histogram(depth_map, "orig depth map", depth_map_histogram_path)
    get_depth_map_img(depth_map_normalized).save(depth_map_normalized_image_path)
    save_histogram(depth_map_normalized, "normalized depth map", depth_map_normalized_histogram_path)

    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    with torch.autocast("cuda"), torch.inference_mode():
        if depth2img_pipe is None:
            raise Exception("Model not loaded, have you set the load_model global variable?")

        image = depth2img_pipe(
            prompt=prompt,
            depth_map=to_depthmap_tensor(depth_map_normalized),
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

    response = make_response(send_file(image_io, mimetype='image/png'))
    response.headers['X-Request-Id'] = request_id

    if order is not None:
        response.headers['X-Order'] = order

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
