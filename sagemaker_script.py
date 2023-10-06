import sagemaker
from sagemaker.pytorch import PyTorch

role = 'SageMaker-training-training-ML-2'
data_location = 's3://sagemaker-studio-mk6unewb9tb/training_data/'

estimator = PyTorch(entry_point='transformer_script.py',
                    role=role,
                    framework_version='2.0.1',  # or your desired version
                    py_version='py310',
                    instance_count=1,
                    instance_type='ml.p3.8xlarge',  # choose a GPU instance type
                    )


estimator.fit(data_location)