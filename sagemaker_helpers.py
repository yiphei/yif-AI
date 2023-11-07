import sagemaker
from sagemaker.pytorch import PyTorchModel
import boto3
from sagemaker.pytorch import PyTorch
from datetime import datetime
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from botocore.config import Config

DEFAULT_ROLE = 'arn:aws:iam::252201027045:role/service-role/SageMaker-training-training-ML-2'

def create_sagemaker_session():
    custom_config = Config(
        read_timeout=600,  # Increase read timeout to 120 seconds (default is 60 seconds)
        retries={'max_attempts': 0}  # Optionally disable retries or adjust max attempts
    )
    my_region = 'us-east-1'  # change to your desired region

    # Creating the sagemaker client using boto3
    sagemaker_client = boto3.client('sagemaker', region_name=my_region)
    sagemaker_runtime_client = boto3.client('sagemaker-runtime', region_name=my_region, config = custom_config)

    # Correcting the default bucket name, it shouldn't be a full S3 path
    default_bucket = 'sagemaker-studio-mk6unewb9tb'  # Ensure this is your correct S3 bucket name

    return sagemaker.Session(default_bucket=default_bucket,
                                        sagemaker_client=sagemaker_client,
                                        sagemaker_runtime_client=sagemaker_runtime_client)


def create_model(model_artifact, model_name, instance_type, sagemaker_session, role = DEFAULT_ROLE):
    pytorch_model = PyTorchModel(model_data=model_artifact,
                                 role = role,
                                framework_version='1.9',  # replace with your PyTorch version
                                py_version='py38',
                                entry_point='inference.py',  # specify your inference script
                                sagemaker_session=sagemaker_session)

    # Now, let's register this model with SageMaker
    sagemaker_session.create_model(name= model_name, role=role, container_defs=pytorch_model.prepare_container_def(instance_type))


def train_model(hyperparameters , instance_type, training_data_s3_path, sagemaker_session, create_model_kwargs = None, role = DEFAULT_ROLE):
    pytorch_estimator = PyTorch(
                                sagemaker_session=sagemaker_session,
                                entry_point='sagemaker_training_script.py', # the name of your script
                                role=role,
                                framework_version='1.9', # select your PyTorch version
                                instance_count=1,
                                instance_type=instance_type, # choose a suitable instance type
                                py_version='py38',
                                hyperparameters=hyperparameters)

    # Now, we'll start a training job.
    pytorch_estimator.fit({'train': training_data_s3_path})

    if create_model_kwargs is not None:
        pytorch_model = pytorch_estimator.create_model(entry_point = 'inference.py')
        sagemaker_session.create_model(name=create_model_kwargs["model_name"],role=role, container_defs=pytorch_model.prepare_container_def(create_model_kwargs["instance_type"]))


def run_transform_job(input_data_s3_path, model_name, instance_type, sagemaker_session):
    transformer = sagemaker.transformer.Transformer(
        model_name=model_name,
        instance_count=1, 
        instance_type=instance_type, 
        strategy='SingleRecord',
        assemble_with='Line',
        output_path=f's3://sagemaker-studio-mk6unewb9tb/inference_output/transform/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
        sagemaker_session=sagemaker_session,
    )

    transformer.transform(data=input_data_s3_path, content_type='application/json', split_type='Line', model_client_config = {"InvocationsTimeoutInSeconds": 3600})

    # Wait for the transform job to finish
    transformer.wait()


def call_async_endpoint(input_data_s3_path, endpoint_name, sagemaker_session):
    return sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async(EndpointName=endpoint_name,
                                    ContentType='application/json',  # or the relevant content type for your data
                                    #   Accept='application/json',
                                    InputLocation=input_data_s3_path
                                    )

def call_endpoint(input_data, endpoint_name, sagemaker_session,):
    predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
)
    prediction = predictor.predict(data = input_data)
    return prediction.decode('utf-8')