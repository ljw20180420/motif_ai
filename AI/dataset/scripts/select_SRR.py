#!/usr/bin/env python

import os
import pathlib
import shutil

import pandas as pd

df = pd.read_csv("assess_SRR/criteras.csv", header=0)
df["meanSignalValueRank"] = df.groupby("accession")["meanSignalValue"].rank(
    "first", ascending=False
)
df = df.query("meanSignalValueRank == 1").reset_index(drop=True)
df.loc[df["accession"] == "O88286", "srr"] = "SRR14790278.sorted.narrowPeak"

data_dir = pathlib.Path(os.environ["DATA_DIR"])
os.makedirs(data_dir / "single", exist_ok=True)
for accession, srr in zip(df["accession"], df["srr"]):
    shutil.copyfile(
        data_dir / "splited" / accession / srr,
        data_dir / "single" / f"{accession}.sorted.narrowPeak",
    )
