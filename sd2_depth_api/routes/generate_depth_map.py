from __main__ import app
import sys
import os
import torch
import uuid
from flask import request, Response
from sd2_depth_api.midas import read_image, infer_depth
from sd2_depth_api.app import logger
from sd2_depth_api.image import load_image
from sd2_depth_api.depth_map import upload_depth_map
from midas.model_loader import default_models, load_model

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

midas_model_path = sys.argv[2]

model_type = midas_model_path.split("/")[-1:][0][:-3]
model_weights = default_models[model_type]

inputs_dir = f"{os.getcwd()}/depth_generation_inputs"

if not os.path.exists(inputs_dir):
    os.mkdir(inputs_dir)

logger.info("Loading MiDaS model")

device = torch.device("cuda")

model, transform, net_w, net_h = load_model(device, midas_model_path, model_type, optimize=False, height=None, square=False)

@app.route('/generate-depth-map', methods=['POST'])
def generate_depth_map():
    request_id = str(uuid.uuid4())
    params = request.json

    if params is None:
        params = dict()

    params_logging = params.copy()
    del params_logging["base_image"]

    logger.info("Depth map generation requested", extra={**{"request_id": request_id}, **params_logging})

    if "base_image" not in params:
        return Response("base_image must be provided", status=400)

    depth_map_url = params['depth_map_url']

    base_image_path = f"{inputs_dir}/{request_id}.png"

    base_image = load_image(params['base_image'])
    base_image.save(base_image_path)

    original_image_rgb = read_image(base_image_path)  # in [0, 1]
    image = transform({"image": original_image_rgb})["image"]

    with torch.no_grad():
        depth_map = infer_depth(device, model, image, original_image_rgb.shape[1::-1])

    logger.info("Depth map generated", extra={**{"request_id": request_id}, **params_logging})

    upload_depth_map(depth_map_url, depth_map)

    logger.info("Depth map uploaded to S3", extra={**{"request_id": request_id}, **params_logging})

    return {
        "request_id": request_id,
    }
