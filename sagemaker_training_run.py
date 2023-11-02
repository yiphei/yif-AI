import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Downloader
import boto3
import os

role = "arn:aws:iam::252201027045:role/service-role/SageMaker-training-training-ML-2"

my_region = 'us-east-1'  # change to your desired region

# Creating the sagemaker client using boto3
sagemaker_client = boto3.client('sagemaker', region_name=my_region)
sagemaker_runtime_client = boto3.client('sagemaker-runtime', region_name=my_region)

# Correcting the default bucket name, it shouldn't be a full S3 path
default_bucket = 'sagemaker-studio-mk6unewb9tb'  # Ensure this is your correct S3 bucket name

sagemaker_session = sagemaker.Session(default_bucket=default_bucket,
                                      sagemaker_client=sagemaker_client,
                                      sagemaker_runtime_client=sagemaker_runtime_client)

pytorch_estimator = PyTorch(
                            sagemaker_session=sagemaker_session,
                            entry_point='sagemaker_training_script.py', # the name of your script
                            role=role,
                            framework_version='1.9', # select your PyTorch version
                            instance_count=1,
                            instance_type='ml.p3.8xlarge', # choose a suitable instance type
                            py_version='py38',
                            hyperparameters={
                                'train_file': 'full_harry_potter.txt',
                                'batch_size': 64,
                                'block_size': 256,
                                'n_embed': 384,
                                'training_steps': 5000,
                                'est_interval': 500,
                                'est_steps': 200,
                                'transform_blocks': 6,
                                'lr': 3e-4,
                                'dropout': 0.2,
                                'n_head': 6
                                # add other hyperparameters you want to pass
                            })

# Now, we'll start a training job.
pytorch_estimator.fit({'train': 's3://sagemaker-studio-mk6unewb9tb/training_data/full_harry_potter.txt'})

job_name = pytorch_estimator.latest_training_job.name  # This gets the name of the last training job

# Build the path where the training artifacts are stored
s3_model_path = f"s3://{default_bucket}/{job_name}/output/model.tar.gz"

# Set the local path where you want to download the model
local_path = f"./model_artifacts/{job_name}/"
os.makedirs(local_path, exist_ok=True)  # Create the local directory if it doesn't exist

# Download the model artifact file
S3Downloader.download(s3_model_path, local_path)