# yif-AI

This is a collection of new AI models and modules that I researched

## Setup

Clone the repo and then install the dependencies

```
pip install -r requirements.txt
```

## Run

To run a model, 

```
python -m <model_dir>.training_script --train datasets/<dataset_dir> --config_file <model_dir>/train_configs/<config_filename>
```

For instance,

```
python -m deep_plan_transformer.training_script --train datasets/wikipedia/ --config_file deep_plan_transformer/train_configs/small.py
```

You can find additional args available in [utils/train.py](utils/train.py)