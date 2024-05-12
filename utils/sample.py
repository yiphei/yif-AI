import argparse
from utils.inference import predict_fn, SampleConfig
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_tokens", type=int)
    args = parser.parse_args()
    return args


def sample(model_fn):
    args = parse_arguments()
    model_dir = os.path.dirname(args.model_path)
    model_filename = os.path.basename(args.model_path)
    SAMPLE_CONFIG = SampleConfig(max_tokens=args.max_tokens)
    model = model_fn(model_dir, model_filename)
    predictions = predict_fn(SAMPLE_CONFIG, model)
    print(predictions)

