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
                            instance_type='ml.p4d.24xlarge', # choose a suitable instance type
                            py_version='py38',
                            hyperparameters={
                                'train_file': 'full_harry_potter.txt',
                                'batch_size': 64,
                                'block_size': 1000,
                                'n_embed': 500,
                                'training_steps': 6000,
                                'est_interval': 500,
                                'est_steps': 200,
                                'transform_blocks': 15,
                                'lr': 3e-4,
                                'dropout': 0.2,
                                'n_head': 10
                                # add other hyperparameters you want to pass
                            })

# Now, we'll start a training job.
pytorch_estimator.fit({'train': 's3://sagemaker-studio-mk6unewb9tb/training_data/full_harry_potter.txt'})

pytorch_model = pytorch_estimator.create_model(entry_point = 'inference.py')
sagemaker_session.create_model(name="test-model", role = role, container_defs=pytorch_model.prepare_container_def('ml.c5.9xlarge'))