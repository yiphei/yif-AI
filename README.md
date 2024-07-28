# yif-AI

This repo contains four AI models that I researched: [***Auto-regressive Encoder-Decoder Transformer***](autoregressive_encoder_decoder_transformer/), [***DeepPlan***](deep_plan_transformer/), [***Future Attention***](future_attention/), and [***Learned Dropout***](learned_dropout/). Each model directory contains an extensive technical exposition along with the complete code implementation required to replicate the reported results.

If you don't know which model to start with, let me help you with two rankings: one by performance against baseline and one by theoretical merits.

Performance against baseline ranking:
1. [***DeepPlan***](deep_plan_transformer/)
2. [***Auto-regressive Encoder-Decoder Transformer***](autoregressive_encoder_decoder_transformer/)
3. [***Learned Dropout***](learned_dropout/)
4. [***Future Attention***](future_attention/)

Theoretical merits ranking:
1. [***DeepPlan***](deep_plan_transformer/)
2. [***Learned Dropout***](learned_dropout/)
3. [***Auto-regressive Encoder-Decoder Transformer***](autoregressive_encoder_decoder_transformer/)
4. [***Future Attention***](future_attention/)

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

If you have GPUs, you can use them by prepending `torchrun --standalone --nproc_per_node=<gpu_num> -m` instead of `python -m`. For instance,

```
torchrun --standalone --nproc_per_node=2 -m deep_plan_transformer.training_script --train datasets/wikipedia/ --config_file deep_plan_transformer/train_configs/small.yaml
```
