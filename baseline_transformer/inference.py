# This exists for Sagemaker to load the model and run inference.
from baseline_transformer.model import TransformerModel
from utils.inference import input_fn as base_input_fn
from utils.inference import model_fn as base_model_fn
from utils.inference import output_fn as base_output_fn
from utils.inference import predict_fn as base_predict_fn


def model_fn(model_dir, file_name=None):
    return base_model_fn(model_dir, TransformerModel, file_name)


def input_fn(input_data, content_type):
    return base_input_fn(input_data, content_type)


def predict_fn(input_data, model):
    return base_predict_fn(input_data, model)


def output_fn(prediction, content_type):
    return base_output_fn(prediction, content_type)
