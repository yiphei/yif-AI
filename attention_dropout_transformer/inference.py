from attention_dropout_transformer.model import AttentionDropoutTransformer
from utils.inference import model_fn as base_model_fn, input_fn as base_input_fn, predict_fn as base_predict_fn, output_fn as base_output_fn

def model_fn(model_dir, file_name=None):
    return base_model_fn(model_dir, AttentionDropoutTransformer, file_name)

def input_fn(input_data, content_type):
    return base_input_fn(input_data, content_type)

def predict_fn(input_data, model):
    return base_predict_fn(input_data, model)

def output_fn(prediction, content_type):
    return base_output_fn(prediction, content_type)