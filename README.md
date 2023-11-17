# Educated Bootstrap Guesser Training

## Description

This repository contains the data and code used to train and evaluate EBG.


## Data
### Data Overview
```
├── data
│   └── comparison_results => contains all cpu/wall-clock time comparison data
│       ├── cpu_times
│       │   └── test_times.csv
│       └── wall_clock_times 
│           ├── benchmark_ebg.csv
│           ├── benchmark_rapid_bootstrap.csv
│           ├── benchmark_raxmlng_ml_searches.csv
│           ├── benchmark_sbs.csv
│           └── benchmark_times_ufboot2.csv
├─────── models => final models of EBG
├─────── processed => computed features, bootstrap targets and training dataset
│        ├── features
│        ├── final
│        └── target 
└─────── raw => all datasets (MSA, tree, model file and bootstrap results) as .tar.gz-file      
```
## Running EBG Training
### 0. Decompression
**Precomputed Training Datatset**
_________________________________________________________________________________________________________________________________________________________________________________________
The current training dataset is stored in [/data/processed/final/final.tar.gz](./data/processed/final/final.tar.gz) as tarball. For further processing ```cd``` into the data directory and perform ```tar -xf final.tar.gz```.\
With this dataset you can directly move to step #4. Training.\
**From Scratch**
_________________________________________________________________________________________________________________________________________________________________________________________
If you want to calculate the features and targets from scratch you need to decompress the raw data.\
The raw datasets used for training and testing at /data are compressed in a tar.gz-file. For further processing ```cd``` into the data directory and perform ```tar -xf datasets.tar.gz```.\
This creates the subdirector /data/raw with all dataset folders including the raw data (MSA, tree file, model file, bootstrap files) which will be used for feature and target calculation.
### 1. Feature Generation
**Precomputed Features**
If you want to use precomuted features you can find a tarball /data/processed/features/features.tar.gz. You have to decompress it using ```cd``` into the data directory and perform ```tar -xf fetures.tar.gz```.\
**From Scratch**
First, you need to ```cd``` into the /features folder. Then perform ```python get_features.py``` from the command line. Since this is computing over 1400 sets of features, it might take a while.\
The code creates a folder for each dataset /data/processed/features including all temporary data as well as the final feature dataset features.csv. Furthermore, after computing all datasets, the features.csv at /data/processed/features contains the training features of all datasets.
### 2. Target Calculation
```cd``` into /scripts and perform ```python target_processing.py```. This will create a file data/processed/target/branch_supports.csv which is the ground truth for the training.
### 3. Training Dataset Creation
```cd``` into /scripts and perform ```python merge_datasets.py```. This will create the final training dataset /data/processed/final/final.csv out of the features and the targets.
### 4. Training


### References
* A. M. Kozlov, D. Darriba, T. Flouri, B. Morel, and A. Stamatakis (2019) 
**RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference** 
*Bioinformatics*, 35(21): 4453–4455. 
[https://doi.org/10.1093/bioinformatics/btz305](https://doi.org/10.1093/bioinformatics/btz305)
