from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, r2_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from skelm import ELMRegressor
import lightgbm as lgb
from datetime import datetime
from lightgbm import early_stopping

from lightgbm import LGBMRegressor
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
# sys.path.insert(0, "..")
import os
import time
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *
from block_split import block_split

# print the JS visualization code to the notebook
shap.initjs()
"""
All the following methods will have to be implemented for the algorithm to work properly
with the same inputs, except for the constructor
"""

pred_dataframe = pd.read_csv('combined_data6_tel12_t_4.csv', index_col=None)
y = pred_dataframe.iloc[:, -1]
X = pred_dataframe.iloc[:, :-1]

n = len(X)
dev_idx, test_idx, test_gap_idx = block_split(n, block_size=50, test_fraction=0.2, gap=10)

n_dev = len(dev_idx)
train_rel_idx, valid_rel_idx, valid_gap_rel_idx = block_split(n_dev, block_size=50, test_fraction=0.25, gap=10)
train_idx = dev_idx[train_rel_idx]
valid_idx = dev_idx[valid_rel_idx]

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]
y_test = y.iloc[test_idx]

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)

# Train the model
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
}

model = lgb.train(
    params,
    d_train,
    num_boost_round=10000,
    valid_sets=[d_valid],
    callbacks=[early_stopping(stopping_rounds=50)]
)


explainer = shap.TreeExplainer(model)
shap_values = explainer(X_valid)           

"""
All the following methods will have to be implemented for the algorithm to work properly
with the same inputs, except for the constructor 
"""
class FS(AbsObjectiveFunc):
    """
    This is the constructor of the class, here is where the objective function can be setted up.
    In this case we will only add the size of the vector as a parameter.
    """
    def __init__(self, size):
        self.size = size
        self.opt = "min" # it can be "max" or "min"
        self.sup_lim = np.full(size, 1) # array where each component indicates the maximum value of the component of the vector
        self.inf_lim = np.full(size, 0) # array where each component indicates the minimum value of the component of the vector
        self.fit1 = None

        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)
    
    """
    This will be the objective function, that will recieve a vector and output a number
    """
    def objective(self, solution):
        X_train_selected = X_train.iloc[:, solution.astype(bool)]
        
        reg = LGBMRegressor(min_data_in_leaf=15, verbosity=-1)
        reg.fit(X_train_selected, y_train)

        X_valid_selected = X_valid.iloc[:, solution.astype(bool)]
        valid_pred = reg.predict(X_valid_selected)

        E = (valid_pred - y_valid)**2
        E = E.mean()

        sign_diff = np.sign(valid_pred - y_valid)
        selected_features = solution.astype(bool)

        P = (shap_values.values[:, selected_features] *
             sign_diff.to_numpy()[:, np.newaxis]).sum(axis=1)   
        P = P/(np.abs(shap_values.values[:, selected_features]).sum(axis=1)+1e-8)  
        P = P.mean()  
        
        baseline_pred = np.full_like(y_valid, y_valid.mean())
        baseline_mse = ((baseline_pred - y_valid)**2).mean()

        E_norm = E / (baseline_mse + 1e-8)

        lambda_shap = 0.1
        fit1 = E_norm + lambda_shap * P

        if self.fit1 is None or fit1 > self.fit1:
            self.fit1 = fit1
        fit1_norm = fit1 / self.fit1
        
        fit2 = np.sum(solution.astype(bool))
        fit2_norm = fit2 / X.shape[1]
       
        w1 = 0.2
        w2 = 1 - w1
        fit = w1 * fit1_norm + w2 * fit2_norm
                  
        return fit

    """

    This will be the function used to generate random vectors for the initializatio of the algorithm
    """
    def random_solution(self):
        return np.random.randint(0, 2, self.size)
    
    """
    This will be the function that will repair solutions, or in other words, makes a solution
    outside the domain of the function into a valid one.
    If this is not needed simply return "solution"
    """
    def repair_solution(self, solution):
        return np.clip(solution, self.inf_lim, self.sup_lim)
   

params = {
    "popSize": 100,
    "rho": 0.6,
    "Fb": 0.98,
    "Fd": 0.2,
    "Pd": 0.8,
    "k": 3,
    "K": 20,
    "group_subs": True,

    "stop_cond": "Neval",
    "time_limit": 4000.0,
    "Ngen": 10000,
    "Neval": 200000,
    "fit_target": 1000,

    "verbose": True,
    "v_timer": 1,
    "Njobs": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 10,
    "prob_amp": 0.01
}
    

operators = [
    SubstrateInt("BLXalpha", {"F":0.8}),
    SubstrateInt("Multipoint"),
    SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
    SubstrateInt("Xor"),
    #SubstrateInt("Spearman", {"X": X, "threshold": 0.95}),  
]

def create_unique_directory(base_dir):
    """Crea una cartella con un nome unico per salvare i risultati."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    else:
        i = 1
        new_dir = f"{base_dir}_{i}"
        while os.path.exists(new_dir):
            i += 1
            new_dir = f"{base_dir}_{i}"
        os.makedirs(new_dir)
        base_dir = new_dir
    return base_dir

base_dir = "results_tele_shap"


# Multiple Runs
tic_total = time.perf_counter()
for i in range(1):
    tic_run = time.perf_counter()
    objfunc = FS(X.shape[1])

    # base_dir = "results_tele_shap"
    output_dir = create_unique_directory(base_dir)

    cro_alg = CRO_SL(objfunc, operators, params)
    solution, obj_value = cro_alg.optimize()

    original_dir = os.getcwd()

    cro_alg.save_solution(os.path.join(output_dir, "solution.csv"))

    solution_file = os.path.join(output_dir, "best_solution.csv")
    population_file = os.path.join(output_dir, "last_population.csv")
    history_file = os.path.join(output_dir, "fit_history.csv")
    prob_file = os.path.join(output_dir, "prob_history.csv")
    indiv_history = os.path.join(output_dir, "indiv_history.csv")

    cro_alg.save_data(
        solution_file=solution_file,
        population_file=population_file,
        history_file=history_file,
        prob_file=prob_file,
        indiv_history=indiv_history
    )

    try:
        os.chdir(output_dir)

        figure_path = "fig.eps" 
        cro_alg.display_report(show_plots=True, save_figure=True, figure_name=figure_path)

    finally:
        os.chdir(original_dir)



    force_plot_path = os.path.join(output_dir, "pycrosl.png")
    shap.force_plot(explainer.expected_value, shap_values.values[1, :], X_valid.iloc[0, :])
    plt.savefig(force_plot_path, format='png')  
    plt.close() 


    summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    shap.summary_plot(shap_values, X_valid)
    plt.savefig(summary_plot_path, format='png') 
    plt.close() 

    toc_run = time.perf_counter()
    print(f"Run {i+1}/1 completato in {toc_run - tic_run:.2f} secondi")

toc_total = time.perf_counter()
print(f"\nTempo totale: {toc_total - tic_total:.2f} secondi ({(toc_total - tic_total)/60:.1f} minuti)")
