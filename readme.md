# Positive unlabeled learning-based enzyme promiscuity prediction

This repository contains the positive unlabeled learning-based enzyme promiscuity prediction (PU-EPP) model as described in the paper Deep learning Enables Rapid Identification of Mycotoxin-degrading Enzymes.

# Requirements
A linux workstation with GPUs is essential for deploying PU-EPP. The final PU-EPP model was trained on 5 NVIDIA Tesla V100 GPUs, which took about 2 weeks.

# Installation

## Dependency
The code has been tested in the following environment:

|  Package    | Version  |
|  ----  | ----  |
| Python  | 3.9.12 |
| PyTorch  | 1.12.0 |
| CUDA  | 11.6.1 |
| RDKit  | 2022.3.5 |
| Gensim  | 4.1.2 |
| Scikit-learn | 1.1.3 |

# Install dependencies
Install [anaconda](https://www.anaconda.com/) first, then install the conda environment by:

```bash
conda env create -f PU_EPP_environment.yml
conda activate PU_EPP
pip install --upgrade pip
pip install jupyter
jupyter notebook
```

# Training
Run train.ipynb and specify `--class CFG` to your own config.

# Testing
Run test.ipynb and specify `--class CFG` to your own config.

# Predicting
## Screening catalytic enzymes for a substrate from a .faste file
1. To load PU-EPP model and make predictions from a .faste file, put the ***example1.fasta*** (.fasta file of candidate enzymes) in the ***data*** folder, 
2. Run predict.ipynb and specify:
* `--PreCFG.useFasteFile`  = True
* `--PreCFG.fasteFile` Path to the .faste file of candidate enzymes
* `--PreCFG.compound`  The molecular structure of the substrate in simplified molecular input line entry system (SMILES) format



After a few minutes of calculation, you will find the screening result in the **result** folder with the name of ***example1_result.csv***.

## Predicting the probes of enzyme-substrate pairs
1. To load PU-EPP and make predictions for enzyme-substrate pairs, put the ***example2.csv*** file (data of enzyme-substrate pairs) in the ***data*** folder.

    ***example2.csv*** 
    |  Compound    | Protein  |
    |  ----  | ----  |
    | SMILES1  | SEQ1 |
    | SMILES2  | SEQ2 |

2. Run predict.ipynb and specify:
* `--PreCFG.useFasteFile` = False
* `--PreCFG.csvFile` Path to the .csv file of enzyme-substrate pairs

You will find the result in the **result** folder with the name of ***example2_result.csv***.

# Fine-tuning 

To fine-tune PU-EPP on a new dataset:
1. put the ***example3_train.csv*** file (data of enzyme-substrate pairs) and ***example3_test.csv*** file in the ***data*** folder.

    ***example3_test.train*** or ***example3_test.csv***

    |  Compound    | Protein  | Label |
    |  ----  | ----  | ---- |
    | SMILES1  | SEQ1 | 1 |
    | SMILES2  | SEQ2 | 0 |
    
2. Run finetuning.ipynb and specify:
* `--CFG.traindata_path` Path to the .csv file of training set
* `--CFG.testdata_path` Path to the .csv file of test set
* `--CFG.modelsave_file_suffix` The suffix of the model name to save
* `--CFG.result_file_suffix` The suffix of the log file name to save

You will find the fine-tuned model named with `--CFG.modelsave_file_suffix` as a suffix in the ***model/model_funetuning*** folder.


# Assistance
For researchers do not have the hardware to deploy PU-EPP, please send your data in one of following formats to us (dachuan.zhang@ifu.baug.ethz.ch or qnhu@sibs.ac.cn), and we'll get the results back to you ASAP.

1)a substrate and a list of candidate enzymes

2)a enzymes and a list of candidate substrates

3)a list of enzyme-substrate pairs

# Link to other repositories
Zenodo, https://doi.org/10.5281/zenodo.7813738

