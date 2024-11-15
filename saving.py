import os
import json
import argparse
import torch
import random
import glog
from utils import BitnetTokenizer, BitLinear, BitLinearNew, BitnetForCausalLM, weight_quant_new
from tqdm import tqdm
import bitnet_cpp
import torch.nn as nn
from pathlib import Path
import io
import numpy as np
from utils import BitLinear
import math
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='1bitLLM/bitnet_b1_58-3B', type=str)
parser.add_argument("--output_path", default='checkpoint', type=str)

def replace_bitlinear_with_new(model):
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            quantized_weight, scale = weight_quant_new(module.weight)
            new_layer = BitLinearNew(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                scale=scale,
                init_weight=quantized_weight,
                input_bits=8
            )
            setattr(model, name, new_layer)
        else:
            replace_bitlinear_with_new(module)
    return model

def save_model_config(model, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    config = model.config if hasattr(model, 'config') else model.module.config
    with open(save_dir / "config.json", 'w') as f:
        json.dump(vars(config), f, default=str)

def split_save_model_state(model, save_dir, chunk_size_mb=90):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Save full model state temporarily
    temp_path = save_dir / "temp_model.pt"
    torch.save(model.state_dict(), temp_path)
    
    # Split using Unix split command
    subprocess.run([
        'split', 
        '-b', f'{chunk_size_mb}m',
        str(temp_path),
        str(save_dir / 'model.pt.part_')
    ])
    
    # Clean up temp file
    os.remove(temp_path)

def main(args):
    model_str = args.hf_path
    model = BitnetForCausalLM.from_pretrained(
        args.hf_path,
    )
    save_model_config(model, args.output_path)
    
    # Replace all BitLinear layers with BitLinearStatic
    model = replace_bitlinear_with_new(model)
    glog.info('Model ready!')
    split_save_model_state(model, args.output_path)
    glog.info('Model saved!')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)