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


# Create a PyTorch model from the trained artifact
model_artifact = 's3://sagemaker-studio-mk6unewb9tb/pytorch-training-2023-10-30-23-11-36-961/output/model.tar.gz'  # replace with the path to your trained model artifact
pytorch_model = PyTorchModel(model_data=model_artifact,
                             role=role,
                             framework_version='1.9',  # replace with your PyTorch version
                             py_version='py38',
                             entry_point='sagemaker_inference.py',  # replace with your inference script
                             sagemaker_session=sagemaker_session)

image_uri = image_uris.retrieve(
    framework='pytorch',
    region=my_region,
    version='1.9',
    py_version='py38',
    instance_type='ml.p3.8xlarge',  # specify the instance type used for training
    image_scope='inference'  # 'inference' or 'training' based on the context
)

model_name = 'yifei-model'
create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_artifact,
        'Environment': pytorch_model.env  # or other environment variables you want to set
    }
)

endpoint_config_name = 'inference-endpoint-config'

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,  # This is obtained from the PyTorchModel object after initialization
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.p3.8xlarge',
            'ModelDataDownloadTimeoutInSeconds': 3600
        }
    ]
)

# async_config = AsyncInferenceConfig(output_path="s3://sagemaker-studio-mk6unewb9tb/inference_output/")

# Deploy the model for asynchronous inference
# predictor = pytorch_model.deploy(endpoint_config_name = endpoint_config_name,
#                                  endpoint_name='inference-endpoint',
#                                 #   async_inference_config=async_config,
#                                  )


endpoint_name = 'inference-endpoint'

# Create the SageMaker endpoint
create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# pytorch_model.deploy(initial_instance_count=1, instance_type='ml.p3.8xlarge')
