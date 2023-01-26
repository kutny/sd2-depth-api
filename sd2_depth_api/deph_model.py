import torch
from diffusers import StableDiffusionDepth2ImgPipeline, EulerAncestralDiscreteScheduler

def create_model_pipe(model_path):
    scheduler = EulerAncestralDiscreteScheduler.from_config(
        "stabilityai/stable-diffusion-2-depth",
        subfolder="scheduler",
    )

    return StableDiffusionDepth2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        scheduler=scheduler,
    ).to("cuda")
