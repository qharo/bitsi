import os, json, argparse, torch, random, glog
from pathlib import Path
from utils import BitLinear, BitLinearNew, BitnetForCausalLM, BitnetConfig, BitnetTokenizer
from transformers import GenerationMixin

def load_model_with_replacement(model_class, load_dir):
    load_dir = Path(load_dir)
    
    # Create model from config
    config = model_class.config_class()
    for k, v in json.load(open(load_dir / "config.json")).items():
        setattr(config, k, v)
    model = model_class(config)
    glog.info('Model architecture loaded!')
    
    # Replace BitLinear layers
    def replace_layers(m):
        for n, mod in m.named_children():
            if isinstance(mod, BitLinear):
                setattr(m, n, BitLinearNew(
                    mod.in_features, mod.out_features,
                    bias=mod.bias is not None,
                    input_bits=8
                ))
            else:
                replace_layers(mod)

    replace_layers(model)
    
    # Load weights safely
    with open(load_dir / 'temp_model.pt', 'wb') as f:
        for part in sorted(load_dir.glob('model.pt.part_*')):
            f.write(open(part, 'rb').read())
    model.load_state_dict(torch.load(load_dir / 'temp_model.pt', weights_only=True), strict=False)
    os.remove(load_dir / 'temp_model.pt')
    glog.info('Model loaded!')
    
    return model

def main(args):
    # Set seeds
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    with torch.no_grad():
        model = load_model_with_replacement(BitnetForCausalLM, args.model_dir).eval()
        tokenizer = BitnetTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
        glog.info('Tokenizer loaded, generating...')

        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_text[len(args.prompt):]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--hf_path', type=str, default='1bitLLM/bitnet_b1_58-3B')
    parser.add_argument('--model_dir', type=str, default='checkpoint')
    parser.add_argument('--prompt', type=str, default="Hi, my name is")
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.5)
    
    main(parser.parse_args())