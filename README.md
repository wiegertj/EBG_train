# Educated Bootstrap Guesser Training

## Description

This repository contains the data and code used to train and evaluate EBG.


## Data
### Data Overview
```
├── data
│   └── comparison_results 
│       ├── cpu_times => cpu times for comparison
│       │   └── test_times.csv
│       ├── predictions => EBG predictions, ultrafast bootstrap predictions, rapid bootstrap predictions
│       │   ├── ebg_prediction_test.csv  
│       │   ├── iq_boots.csv
│       │   └── raxml_classic_supports.csv
│       ├── simulations => results of all tools fon the simulation data
│       │   ├── alrt_simulations.csv
│       │   ├── ebg_simulation.csv
│       │   ├── ufb_simulations.csv
│       │   └── simu_diffs.csv => Pythia difficulty prediction for all simulation MSAs
│       └── wall_clock_times => wall clock times for all tools
│           └── wallclock_times.csv
├─────── models => final models of EBG
├─────── processed 
│        ├── features => EBG features as .tar.gz-file
│        ├── final => final EBG dataset as .tar.gz-file
│        └── target => final EBG targets as .csv
└─────── datasets.tar.gz => all datasets (MSA, tree, model file and bootstrap results) as .tar.gz-file      
```
# Running EBG Training
## 0. Preparation
### OPTION 1: Use precomputed training data
The current training dataset is stored in [/data/processed/final/final.tar.gz](./data/processed/final/final.tar.gz) as tarball. For further processing ```cd``` into the data directory and perform ```tar -xf final.tar.gz```.\
With this dataset you can directly move to step [4. Training](#4.-training).
### OPTION 2: Calculate everything from scratch
If you want to calculate the features and targets from scratch you need to decompress the raw data.\
The raw datasets used for training and testing at /data are compressed in a tar.gz-file. For further processing ```cd``` into the data directory and perform ```tar -xf datasets.tar.gz```.\
This creates the subdirector /data/raw with all dataset folders including the raw data (MSA, tree file, model file, bootstrap files) which will be used for feature and target calculation.
## 1. Feature Generation
### OPTION 1: Use precomputed features 
If you want to use precomuted features you can find a tarball [/data/processed/features/features.tar.gz](./data/processed/features/features.tar.gz). You have to decompress it using ```cd``` into the data directory and perform ```tar -xf fetures.tar.gz```.
### OPTION 2: Calculate features from scratch
First, you need to ```cd``` into the /features folder. Then perform ```python get_features.py``` from the command line. Since this is computing over 1400 sets of features, it might take a while.\
The code creates a folder for each dataset [/data/processed/features](./data/processed/features) including all temporary data as well as the final feature dataset features.csv. Furthermore, after computing all datasets, the features.csv at /data/processed/features contains the training features of all datasets.
## 2. Target Calculation
```cd``` into [/scripts](./scripts) and perform ```python target_processing.py```. This will create a file data/processed/target/branch_supports.csv which is the ground truth for the training.
## 3. Training Dataset Creation
```cd``` into [/scripts](./scripts) and perform ```python merge_datasets.py```. This will create the final training dataset /data/processed/final/final.csv out of the features and the targets.
## 4. Training
Training EBG Regressor: ```cd``` into [/training](./training) and perform ```python ebg_regressor.py```\
This script trains the 5%/10% lower bound as well as the median prediction.
\
\
Training EBG Classifiers: ```cd``` into [/training](./training) and perform ```python ebg_classifier.py```\
This script sequentially trains four EBG classifiers for the different threshold (0.70, 0.75, 0.80, 0.85).

# Paper Plots
All plots of the paper can be found in [/plots/paper_figures](./plots/paper_figures), the scripts for creating them are part of [/plots](./plots):
```
├── plots
│   ├── paper_figures => contains all plots of the paper
│   ├── ebg_correlation_histogram.py => Creates histogram of EBG Pearson correlations
│   ├── ebg_uncertainty_filter_classifier.py => Creates EBG uncertainty filter plot for the EBG classifier
│   ├── ebg_uncertainty_filter_regressor.py => Creates EBG uncertainty filter plot for the EBG regressor
│   ├── runtime.py => Creates runtime comparison plot
│   ├── simulated_data.py => Creates tool comparison on simulated data
│   └── simulated_data_difficulty.py => Creates Pythia difficulty dependent tool comparison on simulated data
```
_________________________________________________________________________________________________________________________________________________________________________________________
### References
* A. M. Kozlov, D. Darriba, T. Flouri, B. Morel, and A. Stamatakis 
**RAxML-NG: a fast, scalable and user-friendly tool for maximum likelihood phylogenetic inference** 
*Bioinformatics*, 35(21):4453–4455, 2019,
[https://doi.org/10.1093/bioinformatics/btz305](https://doi.org/10.1093/bioinformatics/btz305)

* A. Stamatakis, P. Hoover, and J. Rougemont. A Rapid Bootstrap Algo-
rithm for the RAxML Web Servers. Systematic Biology, 57(5):758–771,
2008, [https://doi.org/10.1080/10635150802429642]()  

* D. T. Hoang, O. Chernomor, A. von Haeseler, B. Q. Minh, and L. S. Vinh.
UFBoot2: Improving the ultrafast bootstrap approximation. Molecular
Biology and Evolution, 35(2):518–522, 2018, [https://doi.org/10.1093/molbev/msx281]()

* J. Haag, D. Hoehler, B. Bettisworth, and A. Stamatakis. From Easy to Hope-
less—Predicting the Difficulty of Phylogenetic Analyses. Molecular Biology
and Evolution, 39(12):msac254, 2022, [https://doi.org/10.1093/molbev/msac254]() 

