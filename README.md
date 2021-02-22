# Learning How to Propagate Messages in Graph Neural Networks
This repository contains a PyTorch implementation of "Learning How to Propagate Messages in Graph Neural Networks".

## Dependencies
- CUDA 10.1
- python 3.6.9
- pytorch 1.3.1
- networkx 2.1
- scikit-learn

## Datasets

The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed), and the `newdata` folder contains four datasets(Chameleon, Cornell, Texas, Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the same semi-supervised setting as [GCN](https://github.com/tkipf/gcn) and the same full-supervised setting as Geom-GCN. PPI can be downloaded from [GraphSAGE](http://snap.stanford.edu/graphsage/).



## Usage

- To replicate the semi-supervised results, run the following script
```sh
sh semi.sh
```
- To replicate the full-supervised results, run the following script
```sh
sh full.sh
```

## Reference implementation
The `PyG` folder includes a simple *PyTorch Geometric* implementation of GCNII.
Requirements: `torch-geometric >= 1.5.0` and  `ogb >= 1.2.0`.
- Running examples
```
python cora.py
python arxiv.py
```