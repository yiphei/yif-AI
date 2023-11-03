import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, 
                    format='YY %(asctime)s [%(levelname)s]: %(message)s')
logger.setLevel(logging.INFO) 

import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import json

class OptimizedMultiAttentionHead(nn.Module):

    def __init__(self, dim_in, n_heads, head_size, dropout, block_size):
        super().__init__()
        self.head_size = head_size
        self.q_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size)  * 0.02)
        self.k_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size) * 0.02)
        self.v_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size)* 0.02)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_in, dim_in)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape # B, T, C
        x_expanded = x.unsqueeze(1) # B, 1, T, C
        q = x_expanded @ self.q_weight # B,H,T,C @ H,C,S ->B, H, T, S
        k = x_expanded @ self.k_weight # B,H,T,C @ H,C,S ->B, H, T, S
        v = x_expanded @ self.v_weight # B,H,T,C @ H,C,S ->B, H, T, S

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # B,H,T,S @ B,H,S,T ->B, H, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # B,H,T,T @ B,H,T,S -> B,H,T,S
        B,H,T,S = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, T, S*H) # B,H,T,S -> B,T,H,S -> B,T,H*S
        out = self.proj(out)
        out = self.dropout_2(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, dim_in, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_in * 4),
            nn.ReLU(),
            nn.Linear(dim_in * 4, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_heads, dropout, block_size):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_multi_head = OptimizedMultiAttentionHead(n_embed, n_heads, head_size, dropout, block_size)
        self.feed_forward = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_multi_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, token_size, n_embed, block_size, n_head, transform_blocks, device, dropout):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding = nn.Embedding(token_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.transformer_blocks = nn.Sequential( *[TransformerBlock(n_embed, n_head, dropout, block_size) for _ in range(transform_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, token_size)     
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(torch.arange(x.shape[1], device=self.device))
        embed = token_embed + pos_embed
        out = self.transformer_blocks(embed)
        out = self.ln(out)
        logits = self.output_layer(out)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:,-self.block_size:], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x
    

def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def model_fn(model_dir):
    """
    Load the PyTorch model from the specified directory.
    """
    logger.info("YIFEIIIIII: model_fn - START")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assuming the model is saved as `model.pth` in the model directory
    model_file = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
    
    state_dict = model_file["state_dict"]
    hyperparameters = model_file["hyperparameters"]
    itoc = model_file["itoc"]

    model = Transformer(**hyperparameters, device = device)

    # Check if there are multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

    if not (torch.cuda.is_available() and torch.cuda.device_count() > 1):
        state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict)
    model.to(device)

    logger.info("YIFEIIIIII: model_fn - END")
    return {"model": model, "itoc": itoc}


def input_fn(request_body, request_content_type):
    """
    Parse and preprocess the input data.
    """
    logger.info("YIFEIIIIII: input_fn - START")

    if request_content_type == 'application/json':
        payload = json.loads(request_body)
        data = payload.get('data')
        output_length = payload.get('output_length', 4000)  # you can set a default value
        logger.info("YIFEIIIIII: input_fn - END")
        return data, output_length
    raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_args, model_and_itoc):
    """
    Make prediction on the input data using the loaded model.
    """
    logger.info("YIFEIIIIII: predict_fn - START")
    data, output_length = input_args

    model = model_and_itoc["model"]
    itoc = model_and_itoc["itoc"]
    decoder = lambda x: "".join([itoc[i] for i in x])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # Convert input data to tensor
        data_list = json.loads(data)
        data = torch.tensor([data_list], device=device)
        # Get model predictions
        output = model.generate(data, output_length)
    
    logger.info("YIFEIIIIII: predict_fn - END")
    return decoder(output[0].tolist())

def output_fn(prediction, accept):
    logger.info("output_fn - START")
    content_type = 'text/plain'  # This specifies the MIME type of the output
    logger.info("output_fn - END")
    return prediction, content_type