import sys
import os
sys.path.append(os.getcwd())
from flask import Flask
import os
import torch
from sd2_depth_api.app import logger

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

argument_labels = ["[artifacts base path]", "[SD depth model dir]", "[MiDaS model path]"]

if len(sys.argv) != (len(argument_labels) + 1):
    logger.error("Usage: main.py " + ("  ".join(argument_labels)))
    sys.exit(1)

app = Flask(__name__)

logger.info("sd2-depth-api started")

from sd2_depth_api.routes import infer_images, generate_depth_map

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
