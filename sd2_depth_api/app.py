import boto3
from sd2_depth_api.logger import create_logger

logger = create_logger("sd2-depth-api")

s3 = boto3.client("s3")
