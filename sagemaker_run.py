import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = get_execution_role()

pytorch_estimator = PyTorch(entry_point='sagemaker_script.py', # the name of your script
                            role=role,
                            framework_version='1.8.1', # select your PyTorch version
                            instance_count=1,
                            instance_type='ml.p3.8xlarge', # choose a suitable instance type
                            hyperparameters={
                                'batch_size': 64,
                                'block_size': 256,
                                'n_embed': 384,
                                # add other hyperparameters you want to pass
                            })

# Now, we'll start a training job.
pytorch_estimator.fit({'train': 's3://sagemaker-studio-mk6unewb9tb/training_data/full_harry_potter.txt'})
