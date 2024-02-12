import os
import tiktoken
import numpy as np
import argparse
from pathlib import Path
import pickle
from enum import Enum

class EncoderType(Enum):
    TIKTOKEN = "tiktoken"
    CHAR = "char"

def get_encoding_fn(encoder_type: EncoderType, data: str):
    if encoder_type == EncoderType.TIKTOKEN:
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary, enc.n_vocab
    elif encoder_type == EncoderType.CHAR:
        chars = sorted(list(set(data)))
        alphabet_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        def encode(s):
            return [stoi[c] for c in s]
        return encode, alphabet_size
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--encoder", type=EncoderType)
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    encoder_fn, alphabet_size = get_encoding_fn(args.encoder, data)
    train_ids = encoder_fn(train_data)
    val_ids = encoder_fn(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    file_name = Path(args.file).stem
    train_ids.tofile(os.path.join (os.path.dirname(args.file), f'{file_name}_train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(args.file), f'{file_name}_val.bin'))

    meta = {
        'alphabet_size': alphabet_size,
    }
    with open(os.path.join(os.path.dirname(args.file), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)