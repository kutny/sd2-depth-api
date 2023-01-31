from __main__ import app
import torch
import sys
import uuid
from flask import request, Response, send_file, make_response
import io
import os
import random
from sd2_depth_api.app import logger, artifacts_base_dir
from sd2_depth_api.deph_model import create_model_pipe
from sd2_depth_api.image import load_image
from sd2_depth_api.routes.depth_processor import prepare_depth_map_tensor

sd_depth_model_path = sys.argv[2]

inputs_dir = f"{artifacts_base_dir}/inference_inputs"

if not os.path.exists(inputs_dir):
    os.mkdir(inputs_dir)

logger.info(f"Loading SD depth model from {sd_depth_model_path}")

depth2img_pipe = create_model_pipe(sd_depth_model_path)

@app.route('/infer-images', methods=['POST'])
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

    depth_map_url = params['depth_map_url'] if 'depth_map_url' in params else None
    seed = params['seed'] if "steps" in params else random.randint(1000, 9999)
    negative_prompt = params['negative_prompt'] if "negative_prompt" in params else None
    guidance_scale = params['guidance_scale'] if "guidance_scale" in params else 7
    strength = params['strength'] if "strength" in params else 0.75
    steps = params['steps'] if "steps" in params else 20
    order = params['order'] if "order" in params else None
    normalization_expression = params['normalization_expression'] if "normalization_expression" in params else "np.log(depth_map)"

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
        "order": order,
        "normalization_expression": normalization_expression,
    })

    depth_map_tensor = None

    if depth_map_url:
        depth_map_tensor = prepare_depth_map_tensor(depth_map_url, request_dir, normalization_expression)

    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    with torch.autocast("cuda"), torch.inference_mode():
        if depth2img_pipe is None:
            raise Exception("Model not loaded, have you set the load_model global variable?")

        image = depth2img_pipe(
            prompt=prompt,
            depth_map=depth_map_tensor,
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
