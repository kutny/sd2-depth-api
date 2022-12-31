import logging
import base64
import torch
from flask import Flask, request, Response, send_file
import io
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

app = Flask(__name__)

logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)

depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to("cuda")

def load_image(im_b64) -> Image:
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    return Image.open(io.BytesIO(img_bytes))

@app.route('/', methods=['POST'])
def generate_image():
    params = request.json

    if params is None:
        params = dict()

    if "base_image" not in params:
        return Response("base_image must be provided", status=400)

    base_image = load_image(params['base_image'])

    if "prompt" not in params:
        return Response("prompt must be provided", status=400)

    prompt = params['prompt']

    seed = params['seed'] if "steps" in params else 128 # TODO: randomize
    negative_prompt = params['negative_prompt'] if "negative_prompt" in params else None
    guidance_scale = params['guidance_scale'] if "guidance_scale" in params else 7
    strength = params['strength'] if "strength" in params else 0.75
    steps = params['steps'] if "steps" in params else 20
    
    logging.info(f'Steps: {steps}')
    logging.info(f'Seed: {seed}')

    torch.cuda.manual_seed_all(seed)

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
        ).images[0]

    image_io = io.BytesIO()
    image.save(image_io, 'PNG')
    image_io.seek(0)

    return send_file(image_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
