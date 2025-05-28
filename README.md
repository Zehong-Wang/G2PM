# Scalable Graph Generative Modeling via Substructure Sequences (G2PM)

<div align='center'>

[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.6+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)
[![G2PM arxiv](http://img.shields.io/badge/arxiv-2505.16130-yellow.svg)](https://arxiv.org/abs/2505.16130)

<img src="assets/logo.png" width='300'>
</div>

## 📝 Description (TODO)

This is the official implementation of our paper [Scalable Graph Generative Modeling via Substructure Sequences](https://arxiv.org/abs/2505.16130), the self-supervised version of our previous ICML'25 work [GPM](https://arxiv.org/abs/2501.18739). G2PM uses GPM as the backbone to breakthrough the scalability issue inherent in message passing GNNs, achieving excellent scalability with larger models and more data samples. 

### Key Features
- 🔍 Direct learning from graph substructures instead of message passing
- 🚀 Achieve desirable model scalabilty

### Framework Overview

<img src="assets/paradigm.png">
<img src="assets/framework.png">

G2PM's workflow consists of three main steps:

## 🛠️ Installation

### Prerequisites
- CUDA-compatible GPU (24GB memory minimum, 48GB recommended)
- CUDA 12.1
- Python 3.9+

### Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate GPM

# Install DGL
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# Install PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## 🚀 Quick Start (TODO)
The code of G2PM is presented in folder `/G2PM`. You can run `pretrain.py` and specify any dataset to run experiments. To ensure reproducability, we provide hyper-parameters in `config/pretrain.yaml`. You can simply use command `--use_params` to set tuned hyper-parameters. 

### Basic Usage
```bash
# Run with default parameters
python G2PM/pretrain.py --dataset computers --use_params
```

### Supported Tasks & Datasets

1. **Node Classification**
   - `pubmed`, `photo`, `computers`, `arxiv`, `products`, `wikics`, `flickr`.  

2. **Graph Classification**
   - `imdb-b`, `reddit-m12k`, `hiv`, `pcba`, `sider`, `clintox`, `muv`. 

We also provide the interfaces of other widely used datasets in [GPM](https://github.com/zehong-wang/GPM). Please check the datasets in `G2PM/data/pyg_data_loader.py` for details. 


## 🔧 Configuration Options (TODO)

<!-- ### Training Parameters
- `--use_params`: Use tuned hyperparameters
- `--dataset`: Target dataset name
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--split`: Data split strategy (`public`, `low`, `median`, `high`)

### Model Architecture
- `--hidden_dim`: Hidden layer dimension
- `--heads`: Number of attention heads
- `--num_layers`: Number of Transformer layers
- `--dropout`: Dropout rate

### Pattern Configuration
- `--num_patterns`: Number of patterns per instance
- `--pattern_size`: Pattern size (random walk length)
- `--multiscale`: Range of walk lengths
- `--pattern_encoder`: Pattern encoder type (`transformer`, `mean`, `gru`) -->

For complete configuration options, please refer to our code documentation.

## 📂 Repository Structure
```
└── G2PM
    ├── G2PM/             # Main package directory
    │   ├── data/         # Data loading and preprocessing
    │   ├── model/        # Model architectures
    │   ├── task/         # Task implementations
    │   ├── utils/        # Utility functions
    │   ├── pretrain.py   # Pretraining script
    ├── config/           # Configuration files
    ├── assets/           # Images and assets
    ├── data/             # Dataset storage
    ├── patterns/         # Extracted graph patterns
    └── environment.yml   # Conda environment spec
```

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025scalable,
  title={Scalable Graph Generative Modeling via Substructure Sequences},
  author={Wang, Zehong and Zhang, Zheyuan and Ma, Tianyi and Zhang, Chuxu and Ye, Yanfang},
  journal={arXiv preprint arXiv:2505.16130},
  year={2025}
}
```

## 👥 Authors

- [Zehong Wang](https://zehong-wang.github.io/)
- [Zheyuan Zhang](https://jasonzhangzy1757.github.io/)
- [Tianyi Ma](https://tianyi-billy-ma.github.io/)
- [Chuxu Zhang](https://chuxuzhang.github.io/)
- [Yanfang Ye](http://yes-lab.org/)

For questions, please contact `zwang43@nd.edu` or open an issue.

## 🙏 Acknowledgements

This repository builds upon the excellent work from:
- [GPM](https://github.com/zehong-wang/GPM)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://github.com/snap-stanford/ogb)
- [VQ](https://github.com/lucidrains/vector-quantize-pytorch)

We thank these projects for their valuable contributions to the field.
