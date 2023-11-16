# Educated Bootstrap Guesser Training

## Description

This repository contains the data and code used to train and evaluate EBG.


## Data
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
## Code
### 0. Decompression
The raw datasets used for training and testing at data are compressed in a tar.gz-file. For further processing ```cd``` into the data directory and perform ```tar -xf datasets.tar.gz```.
This creates the subdirector data/raw with all dataset folders including the raw data (MSA, tree file, model file, bootstrap files).
### 1. Feature Generation

### 2. Target Calculation

### 3. Training Dataset Creation

### 4. Training


### References
* A. M. Kozlov, D. Darriba, T. Flouri, B. Morel, and A. Stamatakis (2019) 
**RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference** 
*Bioinformatics*, 35(21): 4453–4455. 
[https://doi.org/10.1093/bioinformatics/btz305](https://doi.org/10.1093/bioinformatics/btz305)
