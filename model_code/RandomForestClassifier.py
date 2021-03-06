import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from evaluate_model import evaluate_model

# dataset = sys.argv[1]
def get_pipeline_parameters():
    n_estimators_values = [10, 50, 100, 500]
    min_impurity_decrease_values = np.arange(0., 0.005, 0.00025)
    max_features_values = [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None]
    criterion_values = ['gini', 'entropy']
    random_state = [324089]

    all_param_combinations = itertools.product(n_estimators_values, min_impurity_decrease_values, max_features_values, criterion_values, random_state)
    pipeline_parameters = \
        [{'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'criterion': criterion, 'random_state': random_state}
        for (n_estimators, min_impurity_decrease, max_features, criterion, random_state) in all_param_combinations]
    return pipeline_parameters

def run(dataset, params, resultdir=".", use_params=True):
    pipeline_parameters = {}
    if use_params:
        pipeline_parameters[RandomForestClassifier] = params

    pipeline_components = [RobustScaler, RandomForestClassifier]
    return evaluate_model(dataset, pipeline_components, pipeline_parameters, resultdir=resultdir)
