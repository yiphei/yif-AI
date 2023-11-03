import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker import image_uris

import boto3

# Initialize the SageMaker session and role
role = 'arn:aws:iam::252201027045:role/service-role/SageMaker-training-training-ML-2'  # replace with your SageMaker role ARN

my_region = 'us-east-1'  # change to your desired region

# Creating the sagemaker client using boto3
sagemaker_client = boto3.client('sagemaker', region_name=my_region)
sagemaker_runtime_client = boto3.client('sagemaker-runtime', region_name=my_region)

# Correcting the default bucket name, it shouldn't be a full S3 path
default_bucket = 'sagemaker-studio-mk6unewb9tb'  # Ensure this is your correct S3 bucket name

sagemaker_session = sagemaker.Session(default_bucket=default_bucket,
                                      sagemaker_client=sagemaker_client,
                                      sagemaker_runtime_client=sagemaker_runtime_client)


model_name = 'big-harry-potter-model-cpu-inference'
transformer = sagemaker.transformer.Transformer(
    model_name=model_name,
    instance_count=1, 
    instance_type='ml.c5.9xlarge', 
    strategy='SingleRecord',
    assemble_with='Line',
    output_path='s3://sagemaker-studio-mk6unewb9tb/inference_output/',
    sagemaker_session=sagemaker_session,
)

input_data = "s3://sagemaker-studio-mk6unewb9tb/inference_input/input_2.json"
transformer.transform(data=input_data, content_type='application/json', split_type='Line')

# Wait for the transform job to finish
transformer.wait()