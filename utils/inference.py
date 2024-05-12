# This exists for Sagemaker to load the model and run inference.
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import tiktoken
import torch


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    # NB: "mps" introduces non-deterministic behavior, despite explicitly setting random seeds.
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class SampleConfig:
    max_tokens: int  # maximum number of tokens to generate for each sample
    n_samples: int = 1  # number of samples to draw
    seed: int = 1337
    device: str = field(default_factory=get_default_device)
    dtype: str = field(
        default_factory=lambda: (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    )
    start_tokens: str = "\n"
    compile: bool = False

    @classmethod
    def create_from_config_file(cls, config_file: str):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {
            k.lower(): v for k, v in config_dict.items() if not k.startswith("__")
        }
        return cls(**config_dict)


def model_fn(model_dir, model_cls, file_name=None):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    device = get_default_device()
    if os.path.isfile(os.path.join(model_dir, file_name or "model.pth")):
        model_dict = torch.load(
            os.path.join(model_dir, file_name or "model.pth"), map_location=device
        )
    else:
        model_dict = torch.load(
            os.path.join(model_dir, file_name or "ckpt.pt"), map_location=device
        )

    model = model_cls.init_from_checkpoint(model_dict)
    model.eval()
    model.to(device)
    model_dict = None
    if torch.cuda.is_available():
        model = torch.compile(model)
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    """
    if request_content_type == "application/json":
        # Assuming JSON inputs. Adjust as necessary.
        request_dict = json.loads(request_body)
        return SampleConfig(**{k.upper(): v for k, v in request_dict.items()})
    else:
        # Handle other content-types here or raise an exception
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: SampleConfig, model):
    """
    Generate model predictions.
    """
    torch.manual_seed(input_data.seed)
    torch.cuda.manual_seed(input_data.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in input_data.device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[input_data.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    start_ids = encode(input_data.start_tokens)
    x = torch.tensor(start_ids, dtype=torch.long, device=input_data.device)[None, ...]

    predictions = []
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(input_data.n_samples):
                y = model.generate(x, input_data.max_tokens)
                predictions.append(decode(y[0].tolist()))

    return predictions


def output_fn(prediction_output, accept="application/json"):
    """
    Serialize and prepare the prediction output.
    """
    if accept == "application/json":
        # Convert prediction output to JSON or other formats as needed
        response_dict = {
            f"pred_{str(i)}": pred for i, pred in enumerate(prediction_output)
        }
        response_body = json.dumps(response_dict)
        return response_body
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
