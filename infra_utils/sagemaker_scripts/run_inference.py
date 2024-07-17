import argparse
import os
from distutils.util import strtobool

import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.serializers import JSONSerializer

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint_name", type=str, required=True)
parser.add_argument(
    "--delete_endpoint", type=lambda v: bool(strtobool(v)), required=True
)
args = parser.parse_args()

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

predictor = PyTorchPredictor(
    endpoint_name=args.endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)
result = predictor.predict({"start_tokens": "\n", "max_tokens": 1000})
# The response format depends on the `output_fn` in your inference script
print(result)

if args.delete_endpoint:
    predictor.delete_endpoint(delete_endpoint_config=True)
    predictor.delete_model()
