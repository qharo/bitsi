
## BitNet b1.58
BitNet b1.58 packing weights into 2 bits, enabling 16x model compression.

## Setup
1. Run setup.sh to create a Python environment, install libraries and CPP
```sh
sh setup.sh
source bin/activate
```

2. To run the [1bitLLM/bitnet_b1_58-xl](https://huggingface.co/1bitLLM/bitnet_b1_58-xl) model
```sh
python3 loading.py
```

2. To save a [BitNet model](https://huggingface.co/1bitLLM) model
```sh
python3 saving.py
```

## Further Improvements
While [BitNet.cpp](https://github.com/microsoft/BitNet) is the official reference implementation, research in this area is [evolving quickly](https://arxiv.org/abs/2411.04965v1). [PyTorch's ao](https://github.com/pytorch/ao) library offers promising tools for native Python implementations. Additionally, I plan to train a low-cost translation model using BitNet once GPU resources become available.
