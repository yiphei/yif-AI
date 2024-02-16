import argparse
import os

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorch
from distutils.util import strtobool
from datetime import datetime

import wandb
from transformer_dropout.training_script import TrainConfig

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
parser.add_argument("--config_file", type=str)
parser.add_argument(
    "--instance_type",
    type=str,
    choices=ALL_INSTANCE_TYPES,
)
parser.add_argument("--instance_count", type=int)
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--use_spot", type=lambda v: bool(strtobool(v)), default=False)
args = parser.parse_args()

# Validate config
assert os.path.exists(f"{SOURCE_DIR}{args.config_file}")
_ = TrainConfig.create_from_config_file(f"{SOURCE_DIR}{args.config_file}")

load_dotenv()
role = os.getenv("SAGEMAKER_ROLE")
wandb.sagemaker_auth(path="transformer_dropout")
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

# Annoying that I have to manually create these, but otherwise sagemaker wont allow me to dynamically
# create a checkpoint directory in the training directory
s3 = boto3.client('s3', region_name=my_region)
training_run_dir = f"training_run_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}/"
checkpoint_dir = "checkpoints/"
s3.put_object(Bucket=default_bucket, Key=training_run_dir)
s3.put_object(Bucket=default_bucket, Key= training_run_dir + checkpoint_dir)

pytorch_estimator = PyTorch(
    sagemaker_session=sagemaker_session,
    entry_point="training_script.py",
    source_dir=SOURCE_DIR,
    role=role,
    framework_version="2.1.0",
    instance_count=args.instance_count,  # increase for multi-node distributed training
    instance_type=args.instance_type,
    py_version="py310",
    distribution=(
        {"torch_distributed": {"enabled": True}}
        if args.instance_type in GPU_INSTANCE_TYPES
        else None
    ),
    hyperparameters={
        "train_file": "full_harry_potter_train.bin",
        "val_file": "full_harry_potter_val.bin",
        "config_file": args.config_file,
        "is_local": "False",
    },
    output_path = f"s3://dropout-transformer/{training_run_dir}",
    code_location = f"s3://dropout-transformer/{training_run_dir}/code/", # annoying that this has to be specified
    checkpoint_s3_uri = f"s3://dropout-transformer/{training_run_dir}{checkpoint_dir}",
    use_spot_instances=args.use_spot,
    max_wait = (60 * 60 *24) if args.use_spot else None,
    tags={"notes": args.notes}
)

pytorch_estimator.fit(
    {"train": "s3://dropout-transformer/datasets/full_harry_potter/"}
)  # add wait=False if you want to run asynchronously

# pytorch_model = pytorch_estimator.create_model(entry_point = 'inference.py')
# sagemaker_session.create_model(name="test-model", role = role, container_defs=pytorch_model.prepare_container_def('ml.c5.9xlarge'))
