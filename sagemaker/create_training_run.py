import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os
from dotenv import load_dotenv
import os

load_dotenv()
role = os.getenv('SAGEMAKER_ROLE')

my_region = 'us-east-1'  # change to your desired region

# Creating the sagemaker client using boto3
sagemaker_client = boto3.client('sagemaker', region_name=my_region)
sagemaker_runtime_client = boto3.client('sagemaker-runtime', region_name=my_region)

# Correcting the default bucket name, it shouldn't be a full S3 path
default_bucket = 'dropout-transformer'  # use env

sagemaker_session = sagemaker.Session(default_bucket=default_bucket,
                                      sagemaker_client=sagemaker_client,
                                      sagemaker_runtime_client=sagemaker_runtime_client)

pytorch_estimator = PyTorch(sagemaker_session=sagemaker_session,
                            entry_point='training_script.py', # the name of your script
                            source_dir='transformer_dropout/',
                            role=role,
                            framework_version='2.1', # select your PyTorch version
                            instance_count=1,
                            instance_type='ml.p4d.24xlarge', # choose a suitable instance type
                            py_version='py310',
                            hyperparameters={
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
pytorch_estimator.fit({'train': 's3://dropout-transformer/datasets/full_harry_potter/full_harry_potter_train.bin'})

pytorch_model = pytorch_estimator.create_model(entry_point = 'inference.py')
sagemaker_session.create_model(name="test-model", role = role, container_defs=pytorch_model.prepare_container_def('ml.c5.9xlarge'))