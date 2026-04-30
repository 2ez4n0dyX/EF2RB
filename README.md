# EF2RB
Ensemble of Filter Feature Selection Methods with Ranker Booster for Classification of High-Dimensional IoT Intrusion Data is a scalable feature selection framework designed for high-dimensional IoT intrusion detection datasets.
The framework integrates:

  -multi-ranker feature evaluation
  -subset-based dimensionality reduction
  -iterative refinement
  -consensus-driven feature selection
  -correlation-based redundancy removal

It is designed to improve feature stability, robustness, and scalability while maintaining strong classification performance.

# Functions.py 
All the Python functions used in this project.

# Feature_Rankers.py
All the ranking functions used in this project. It also has some extra functions other than those used.

# UNSW-NB15 Folder
Main_UNSWNB15.ipynb -- The main code for the UNSW-NB15 dataset.
UNSW-NB15_full.log -- Log files.
UNSW-NB15_full_correlation_full.png -- Correlation of full features before applying our method
UNSW-NB15_full_correlation_optimal.png -- Correlation of optimal features after applying our method.

# Config and hyperparameters for UNSW-NB15 Dataset
num_subsets = 4
random_state=42
n_features= 6
min_ranker_agreement=3
threshold=0.8
