import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

# ugly workound to make both Sagemaker, python, and me happy
try:
    # I like to run the script from the project root as a module, so it needs to be relative import
    from .model import EncoderDecoderTransformer, ModelConfig
except ImportError:
    # Sagemaker prob runs the script as a standalone file, so it needs to be an absolute import
    from model import EncoderDecoderTransformer, ModelConfig

import tiktoken


@dataclass
class SampleConfig:
    N_SAMPLES: int  # number of samples to draw
    MAX_TOKENS: int  # maximum number of tokens to generate for each sample
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
    START_TOKEN: str = "\n"
    COMPILE: bool = False

    @classmethod
    def create_from_config_file(cls, config_file: str):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}
        return cls(**config_dict)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    SAMPLE_CONFIG = SampleConfig.create_from_config_file(args.config_file)

    torch.manual_seed(SAMPLE_CONFIG.SEED)
    torch.cuda.manual_seed(SAMPLE_CONFIG.SEED)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in SAMPLE_CONFIG.DEVICE else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[SAMPLE_CONFIG.DTYPE]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model_dict = torch.load(args.model_path, map_location=SAMPLE_CONFIG.DEVICE)
    model_config = ModelConfig(**model_dict["model_config"])
    model = EncoderDecoderTransformer(model_config)
    state_dict = model_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(SAMPLE_CONFIG.DEVICE)
    if SAMPLE_CONFIG.COMPILE:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    start_ids = encode(SAMPLE_CONFIG.START_TOKEN)
    x = torch.tensor(start_ids, dtype=torch.long, device=SAMPLE_CONFIG.DEVICE)[
        None, ...
    ]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(SAMPLE_CONFIG.N_SAMPLES):
                y = model.generate(x, SAMPLE_CONFIG.MAX_TOKENS)
                print(decode(y[0].tolist()))
                print("---------------")
