# inference.py
import torch
from model import DropoutTransformer, ModelConfig
import tiktoken
from contextlib import nullcontext
import json
import os
from dataclasses import dataclass, field

@dataclass
class SampleConfig:
    MAX_TOKENS: int # maximum number of tokens to generate for each sample
    N_SAMPLES: int = 1 # number of samples to draw
    SEED: int = 1337
    DEVICE: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    DTYPE: str = field(
        default_factory=lambda: (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    )
    START_TOKENS: str = "\n"
    COMPILE: bool = False

    @classmethod
    def create_from_config_file(cls, config_file: str):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}
        return cls(**config_dict)

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
    model_config = ModelConfig(**model_dict['model_config'])
    model = DropoutTransformer(model_config)
    state_dict = model_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()    
    model.to(device)
    model_dict = None
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    """
    if request_content_type == 'application/json':
        # Assuming JSON inputs. Adjust as necessary.
        request_dict = json.loads(request_body)
        return SampleConfig(**{k.upper():v for k,v in request_dict.items()})
    else:
        # Handle other content-types here or raise an exception
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data: SampleConfig, model):
    """
    Generate model predictions.
    """
    torch.manual_seed(input_data.SEED)
    torch.cuda.manual_seed(input_data.SEED)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in input_data.DEVICE else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[input_data.DTYPE]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    start_ids = encode(input_data.START_TOKENS)
    x = (torch.tensor(start_ids, dtype=torch.long, device=input_data.DEVICE)[None, ...])

    predictions = []
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(1):
                y = model.generate(x, input_data.MAX_TOKENS)
                predictions.append(decode(y[0].tolist()))

    return predictions

def output_fn(prediction_output, accept='application/json'):
    """
    Serialize and prepare the prediction output.
    """
    print("YIFEII - OUTPUT_FN")
    if accept == 'application/json':
        # Convert prediction output to JSON or other formats as needed
        response_dict = {f"pred_{str(i)}": pred for i, pred in enumerate(prediction_output)}
        response_body = json.dumps(response_dict)
        return response_body
    else:
        raise ValueError(f"Unsupported accept type: {accept}")