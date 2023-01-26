import sys
import os
sys.path.append(os.getcwd())
from flask import Flask
import os
from sd2_depth_api.app import logger

if len(sys.argv) != 2:
    logger.error("Missing model path argument")
    sys.exit(1)

app = Flask(__name__)

logger.info("sd2-depth-api started")

from sd2_depth_api.routes import infer_images

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4242)
