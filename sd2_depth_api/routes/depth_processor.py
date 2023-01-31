import numpy as np
from sd2_depth_api.depth_map import download_depth_map, get_depth_map, get_depth_map_img, save_histogram, to_depthmap_tensor

def get_depth_map_artifacts_paths(request_dir):
    return dict(   
        depth_map_path = f"{request_dir}/2-depth_map.npy", 
        depth_map_image_path = f"{request_dir}/3-depth_map.png",
        depth_map_histogram_path = f"{request_dir}/4-depth_map_histogram.pdf",
        depth_map_normalized_image_path = f"{request_dir}/5-normalized_depth_map.png",
        depth_map_normalized_histogram_path = f"{request_dir}/6-normalized_depth_map_histogram.pdf"
    )

def log_depth_map(depth_map: np.ndarray, depth_map_normalized: np.ndarray, paths: dict):
    get_depth_map_img(depth_map).save(paths['depth_map_image_path'])
    save_histogram(depth_map, "orig depth map", paths['depth_map_histogram_path'])
    get_depth_map_img(depth_map_normalized).save(paths['depth_map_normalized_image_path'])
    save_histogram(depth_map_normalized, "normalized depth map", paths['depth_map_normalized_histogram_path'])

    return to_depthmap_tensor(depth_map_normalized)

def prepare_depth_map_tensor(depth_map_url: str, request_dir: str, normalization_expression: str):
    depth_map_artifacts_paths = get_depth_map_artifacts_paths(request_dir)
    
    download_depth_map(depth_map_url, depth_map_artifacts_paths['depth_map_path'])
    depth_map, depth_map_normalized = get_depth_map(depth_map_artifacts_paths['depth_map_path'], normalization_expression)

    return log_depth_map(depth_map, depth_map_normalized, depth_map_artifacts_paths)
