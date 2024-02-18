import argparse
import os

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorchModel

SOURCE_DIR = "transformer_dropout/"
GPU_INSTANCE_TYPES = [
    "ml.p3.2xlarge",
    "ml.p3.8xlarge",
    "ml.p3.16xlarge",
    "ml.p3dn.24xlarge",
    "ml.p4d.24xlarge",
]
ALL_INSTANCE_TYPES = GPU_INSTANCE_TYPES + ["ml.c5.18xlarge"]

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint_name", type=str, default=None)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument(
    "--instance_type",
    type=str,
    choices=ALL_INSTANCE_TYPES,
    required=True,
)
parser.add_argument("--model_uri", type=str, required=True)
args = parser.parse_args()


load_dotenv()
role = os.getenv("SAGEMAKER_ROLE")
my_region = "us-east-1"

# Creating the sagemaker client using boto3
sagemaker_client = boto3.client("sagemaker", region_name=my_region)
sagemaker_runtime_client = boto3.client("sagemaker-runtime", region_name=my_region)
default_bucket = "dropout-transformer"

sagemaker_session = sagemaker.Session(
    default_bucket=default_bucket,
    sagemaker_client=sagemaker_client,
    sagemaker_runtime_client=sagemaker_runtime_client,
)

pytorch_model = PyTorchModel(
    role=role,
    name=args.model_name,
    sagemaker_session=sagemaker_session,
    model_data=args.model_uri,  # S3 path to your trained model artifacts
    entry_point="inference.py",  # Your inference script
    source_dir="transformer_dropout/",  # Directory containing your model.py and any other necessary files
    framework_version="2.1.0",  # Version of PyTorch you're using
    py_version="py310",  # Python version
)

pytorch_model.deploy(
    endpoint_name=args.endpoint_name,
    instance_type=args.instance_type,
    initial_instance_count=1,
)
