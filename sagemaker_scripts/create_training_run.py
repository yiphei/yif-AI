import argparse
import os
from datetime import datetime
from distutils.util import strtobool

import boto3
import sagemaker
import wandb
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorch

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


def expression_to_int(expression):
    allowed_chars = set("0123456789*")
    if not all(char in allowed_chars for char in expression.replace(" ", "")):
        raise ValueError("Invalid characters in expression")

    try:
        # Evaluate the expression
        return eval(expression, {"__builtins__": None}, {})
    except (SyntaxError, NameError):
        raise ValueError("Invalid expression")


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument(
    "--instance_type",
    type=str,
    choices=ALL_INSTANCE_TYPES,
    required=True,
)
parser.add_argument("--instance_count", type=int, required=True)
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--use_spot", type=lambda v: bool(strtobool(v)), default=False)
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--output_dir_name", type=str, default=None)
parser.add_argument("--max_runtime", type=lambda exp: expression_to_int(exp))
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
# create a checkpoint directory in the output directory at runtime
s3 = boto3.client("s3", region_name=my_region)

# List objects within the specified S3 prefix
response = s3.list_objects_v2(Bucket=default_bucket, Prefix=args.train)
# Filter for files that end with '_train.bin'
train_files = [
    obj["Key"]
    for obj in response.get("Contents", [])
    if obj["Key"].endswith("_train.bin")
]
val_files = [
    obj["Key"]
    for obj in response.get("Contents", [])
    if obj["Key"].endswith("_val.bin")
]
assert len(train_files) == 1 and len(val_files) == 1

training_run_dir = (
    f"training/sagemaker_training_run_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}/"
    if args.output_dir_name is None
    else f"training/sagemaker_{args.output_dir_name}/"
)
checkpoint_dir = "checkpoints/"
s3.put_object(Bucket=default_bucket, Key=training_run_dir)
s3.put_object(Bucket=default_bucket, Key=training_run_dir + checkpoint_dir)

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
        "config_file": args.config_file,
        "is_local": "False",
    },
    output_path=f"s3://dropout-transformer/{training_run_dir}",
    code_location=f"s3://dropout-transformer/{training_run_dir}/code/",  # annoying that this has to be specified
    checkpoint_s3_uri=f"s3://dropout-transformer/{training_run_dir}{checkpoint_dir}",
    use_spot_instances=args.use_spot,
    max_wait=(60 * 60 * 24) if args.use_spot else None,
    tags={"notes": args.notes},
    # max_run = args.max_runtime, # this is currently not working.
)

pytorch_estimator.fit(
    {"train": f"s3://dropout-transformer/{args.train}"}
)  # add wait=False if you want to run asynchronously
