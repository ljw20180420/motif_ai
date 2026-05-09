#!/usr/bin/env python

import os
import pathlib

import bioframe as bf
import pandas as pd
import py2bit

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

with py2bit.open("genome/mm9.2bit") as tb:
    DNAs = []
    for chrom, start, end in zip(df["chrom"], df["start"], df["end"]):
        DNA = tb.sequence(chrom, start, end)
        DNAs.append(DNA)

df["DNA"] = DNAs

df.to_csv("pCBS_train.csv", index=False)
df[["DNA"]].assign(protein="O88286").to_csv(
    "../../paper/infer_all_models/wiz.csv", index=False
)
