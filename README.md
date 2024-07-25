# yif-AI

This is a collection of new AI models and modules that I researched

## Setup

Clone the repo and then install the dependencies

```
pip install -r requirements.txt
```

## Train

To run a model training, the required args are: the model training script, the dataset, and the training config file. The general pattern is the following

```
python -m <model_dir>.training_script --train datasets/<dataset_dir> --config_file <model_dir>/train_configs/<config_filename>
```

For instance, you can run

```
python -m deep_plan_transformer.training_script --train datasets/wikipedia/ --config_file deep_plan_transformer/train_configs/small.yaml
```

You can find additional training script args in [utils/train.py](utils/train.py)