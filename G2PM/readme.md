# Neural graph pattern machine


## Install environment

You may use conda to install the environment. Please run the following script. 

```
conda env create -f environment.yml
conda activate G2PM
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Run the experiments

You may use the command like

```
python pretrain.py --dataset pubmed --use_params
```

to reproduce the experimental results of the paper.

## Dataset

The datasets can be as follows. 

**Node Classification:** `pubmed`, `photo`, `computers`, `arxiv`, `products`, `wikics`, `flickr`.  

**Graph Classification**: `imdb-b`, `reddit-m12k`, `hiv`, `pcba`, `sider`, `clintox`, `muv`. 