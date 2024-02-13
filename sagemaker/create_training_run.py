import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os
from dotenv import load_dotenv
import os
import wandb

load_dotenv()
role = os.getenv('SAGEMAKER_ROLE')
wandb.sagemaker_auth(path="transformer_dropout")

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
                            instance_type='ml.c5.18xlarge', # choose a suitable instance type
                            py_version='py310',
                            hyperparameters={
                                'train_file': 'full_harry_potter_train.bin',
                                'val_file': 'full_harry_potter_val.bin',
                                'config_file': 'configs/train_debug.py',
                                'is_local': 'False',
                            })

# Now, we'll start a training job.
pytorch_estimator.fit({'train': 's3://dropout-transformer/datasets/full_harry_potter/'})

# pytorch_model = pytorch_estimator.create_model(entry_point = 'inference.py')
# sagemaker_session.create_model(name="test-model", role = role, container_defs=pytorch_model.prepare_container_def('ml.c5.9xlarge'))