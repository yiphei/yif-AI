from sagemaker.predictor import Predictor
import numpy as np
import json
import os
import boto3
import sagemaker
from sagemaker.serializers import JSONSerializer


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

# Specify your endpoint name
endpoint_name = 'aaaaaaa'
# Initialize the predictor
predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, serializer = JSONSerializer())
result = predictor.predict({"instances": [1.0, 2.0, 5.0]})
# The response format depends on the `output_fn` in your inference script
print(result)