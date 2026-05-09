#!/usr/bin/env python

import os
import pathlib

import pandas as pd

os.chdir(pathlib.Path(__file__).resolve().parent.parent)

df = pd.read_csv(
    pathlib.Path(os.environ["DATA_DIR"]) / "train_data" / "O88286.csv", header=0
)
df_pCBS = pd.read_csv("pCBS_train.csv", header=0)
max_oc_num = len(df_pCBS)
print(max_oc_num)

sample_num = 200000
for seed in range(10000):
    df_sample = df.sample(n=sample_num, random_state=seed)
    oc_num = (
        pd.concat([df_sample[["DNA"]], df_pCBS[["DNA"]]]).duplicated().sum()
        - df_sample[["DNA"]].duplicated().sum()
    )
    print(oc_num)
    if oc_num == max_oc_num:
        print(seed)
        break
