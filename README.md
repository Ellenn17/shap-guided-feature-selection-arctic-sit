# Uncovering Drivers of Arctic Sea Ice Thickness Through SHAP-Regularized Evolutionary Feature Selection
PyCROSL + SHAP framework
This repository contains the code and experiments supporting the paper:
Uncovering Drivers of Arctic Sea Ice Thickness Through SHAP-Regularized Evolutionary Feature Selection
E. Poli, A. Iseni, M. Sangiorgio, J. Péerez-Aracil, L. Bianchi, B. Soja, S. Salcedo-Sanz, A. Castelletti

## Overview
Understanding the drivers of Arctic Sea Ice Thickness (SIT) is challenging due to the large number of correlated meteorological, hydrological, and climate predictors involved.
This repository implements an interpretable feature selection framework that combines:

- PyCROSL: an evolutionary optimization algorithm inspired by coral reef dynamics
- SHAP (SHapley Additive exPlanations): used as an internal regularization mechanism to guide feature selection toward coherent and interpretable subsets and embedded directly into the fitness function (not applied post-hoc).

The framework is applied to clustered Arctic regions and evaluates different feature selection strategies under varying constraints.

## Methodological Summary
![The workflow implemented in this repository consists of the following steps:](flowchart.png)
#### 1. Data preprocessing
- Monthly aggregation of heterogeneous datasets (GHCN station meteorology, ArcticGRO river discharge, NOAA CPC teleconnection indices)
- Construction of lagged predictors (1-6 and 1-12 months)
- Standardization and imputation
#### 2. Spatial clustering
PIOMAS sea ice thickness fields clustered into five homogeneous Arctic regions using K-means
#### 3. Feature selection
- PyCROSL (baseline configuration)
- PyCROSL with SHAP-guided fitness
- Each configuration run 5 times (stochastic optimizer); features retained via majority-vote consensus (selected in ≥3/5 runs) after a convergence-screening step based on fitness-history curves
#### 4. Post-selection validation
- Retraining LightGBM models on stable feature subsets
- Independent evaluation of predictive skill (R², correlation, MAE) and comparison against a persistence baseline
- Compared against LASSO (`LassoCV`, 5-fold) and mRMR baselines
## Reproducibility
- `PyCROSL.py` / `PyCROSL_SHAP.py`: main training scripts (plain and SHAP-guided variants)
- `Paper_Figures_and_Validation.ipynb`: consolidated notebook with all functions needed to reproduce the paper's figures (performance table, baseline comparison + heatmap, SHAP importance/dependence plots) and validation numbers (persistence baseline, timing test)

## PyCROSL Dependency

This project builds upon the official **PyCROSL** implementation:

🔗 [https://github.com/GheodeAI/PyCROSL/tree/main/PyCROSL  ](https://github.com/GheodeAI/PyCROSL/tree/main)

The original PyCROSL framework is described in:

Pérez-Aracil, J., Camacho-Gómez, C., Lorente-Ramos, E., Marina, C. M., Cornejo-Bueno, L. M., & Salcedo-Sanz, S. (2023).  
*New probabilistic, dynamic multi-method ensembles for optimization based on the CRO-SL.*  
Mathematics, 11(7), 1666. https://doi.org/10.3390/math11071666

### Important Note

The files contained in the `Algorithm_settings/` directory of this repository do not include the full PyCROSL framework. They only contain a modified version of the main execution script.


To reproduce our results, use commit  of PyCROSL.

