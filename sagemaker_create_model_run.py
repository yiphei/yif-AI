import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker import Model

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


# Create a PyTorch model from the trained artifact
model_artifact = 's3://sagemaker-studio-mk6unewb9tb/pytorch-training-2023-11-02-20-57-14-962/output/model.tar.gz'  # replace with the path to your trained model artifact
pytorch_model = PyTorchModel(model_data=model_artifact,
                             role=role,
                             framework_version='1.9',  # replace with your PyTorch version
                             py_version='py38',
                             entry_point='sagemaker_inference.py',  # specify your inference script
                             sagemaker_session=sagemaker_session)

# Now, let's register this model with SageMaker
sagemaker_session.create_model(name="big-harry-potter-model", role = role, container_defs=pytorch_model.prepare_container_def('ml.p4d.24xlarge'))