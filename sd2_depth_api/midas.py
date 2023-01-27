import torch
import cv2
from sd2_depth_api.app import logger

def read_image(path: str):
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def infer_depth(device, model, image, target_size):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    height, width = sample.shape[2:]
    logger.info(f"Infpu resized to {width}x{height} before entering the encoder")
   
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction
