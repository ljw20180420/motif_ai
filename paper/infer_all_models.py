#!/usr/bin/env python

import pandas as pd

for model_cls in [
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "DecisionTree",
    "CategoricalNB",
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
    "DeepZF",
]:
    df = pd.read_csv(f"paper/infer_all_models/{model_cls}.csv", header=0)
    print(model_cls, df["proba"].max())
