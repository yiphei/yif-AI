import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor

# Initialize the SageMaker session and role
sagemaker_session = sagemaker.Session()
role = 's3://sagemaker-studio-mk6unewb9tb/pytorch-training-2023-10-29-17-41-10-561/output/model.tar.gz'  # replace with your SageMaker role ARN

# Create a PyTorch model from the trained artifact
model_artifact = 's3://sagemaker-studio-mk6unewb9tb/pytorch-training-2023-10-29-04-49-30-583/output/model.tar.gz'  # replace with the path to your trained model artifact
pytorch_model = PyTorchModel(model_data=model_artifact,
                             role=role,
                             framework_version='1.9.0',  # replace with your PyTorch version
                             entry_point='inference.py',  # replace with your inference script
                             sagemaker_session=sagemaker_session)

# Deploy the model for asynchronous inference
predictor = pytorch_model.deploy(initial_instance_count=1,
                                 instance_type='ml.p3.8xlarge',
                                 endpoint_name='async-inference-endpoint',
                                 asynchronous=True)
