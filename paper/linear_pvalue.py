#!/usr/bin/env python

import os
import pathlib

import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress

os.chdir(pathlib.Path(__file__).resolve().parent)

df = pd.read_csv(
    "mouse_rna_distance_bias.txt", sep="\t", skiprows=[0], names=["gene", "x", "y"]
).dropna()

result = linregress(df["x"].to_numpy(), df["y"].to_numpy(), alternative="greater")
print(result)

model = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
model.summary()
