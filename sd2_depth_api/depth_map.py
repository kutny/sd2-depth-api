import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
