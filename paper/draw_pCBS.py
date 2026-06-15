#!/usr/bin/env python

import os
import pathlib

import pandas as pd
import seaborn as sns

os.chdir(pathlib.Path(__file__).resolve().parent.parent)

pCBS_dir = pathlib.Path("paper/pCBS/output")

dfs = []
for model_cls in [
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "DecisionTree",
    "CategoricalNB",
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
    "SupportVectorMachine",
    "DeepZF",
    "COP",
]:
    df = pd.read_csv(pCBS_dir / f"{model_cls}.csv", header=0).assign(
        model_cls=model_cls
    )
    dfs.append(df)

df_count: pd.DataFrame = (
    pd
    .concat(dfs, ignore_index=True)
    .assign(bind=lambda df: df["proba"] >= 0.5)
    .groupby([
        "model_cls",
        "protein",
    ])["bind"]
    .sum()
    .reset_index()
)

os.makedirs("paper/draw_pCBS", exist_ok=True)
sns.clustermap(
    df_count.pivot_table(values="bind", index="protein", columns="model_cls"),
    figsize=(10, 100),
    dendrogram_ratio=(0.2, 0.02),
    annot=True,
).savefig("paper/draw_pCBS/cluster.png")
