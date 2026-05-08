#!/usr/bin/env python

import os
import pathlib

import bioframe as bf
import pandas as pd

os.chdir(pathlib.Path(__file__).resolve().parent.parent)

df_pCBS = pd.read_csv("pCBS.csv", header=0)

wiz_file = pathlib.Path(os.environ["DATA_DIR"]) / "sized" / "O88286.sized.narrowPeak"
df_wiz = pd.read_csv(
    wiz_file,
    sep="\t",
    names=["chrom", "start", "end", "summit"],
)
df = bf.closest(df_pCBS, df_wiz)
df = df[["chrom_", "start_", "end_"]].rename(
    columns={"chrom_": "chrom", "start_": "start", "end_": "end"}
)
df.to_csv("pCBS_train.csv", index=False)
