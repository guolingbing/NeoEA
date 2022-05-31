# Understanding and Improving Knowledge Graph Embedding for Entity Alignment

## Introduction

This repository is the official implementation of *Understanding and Improving Knowledge Graph Embedding for Entity Alignment, ICML 2022*.

Please see the paper [A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf) (VLDB 2020) for dataset details.


## Dependencies

This project is based on [OpenEA](https://github.com/nju-websoft/OpenEA).  We did not add additional packages compared with the original OpenEA project. 

## Example

We provide an example of jupyter [notebook](https://github.com/guolingbing/NeoEA/blob/main/run/main_from_notebook.ipynb).

## Quick Start

### 1. Creat a new conda env and install packages

```bash
conda create -n openea python=3.6
conda activate openea
conda install tensorflow-gpu==1.8
conda install -c conda-forge graph-tool==2.29
conda install -c conda-forge python-igraph
pip install -r requirement.txt
```

### 2. Install the local package

```bash
pip install -e .
```

### 3. Use the same scripts in OpenEA to run a model with NeoEA:

```bash
python main_from_args.py ./args/sea_args_15K.json D_W_15K_V1 721_5fold/1/
```



## The Code Location of NeoEA

### 1. Neural ontology and neural axioms

 ./src/openea/approaches/neural_ontology.py


### 2. Baselines

We slightly modified the source code of the baselines to inject neural ontology into them:

BootEA: ./src/openea/approaches/bootea.py

SEA: ./src/openea/approaches/sea.py

RSN: ./src/openea/approaches/rsn4ea.py

RDGCN: ./src/openea/approaches/rdgcn.py

### 3. Parameter settings

We added NeoEA hyper-parameters to the original setting files:

BootEA: ./run/args/bootea_args_15K.json

SEA: ./run/args/sea_args_15K.json

RSN: ./run/args/rsn4ea_args_15K.json

RDGCN: ./run/args/rdgcn_args_15K.json
