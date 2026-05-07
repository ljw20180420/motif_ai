#!/usr/bin/env python

import os
import pathlib

import pandas as pd

os.makedirs("AI/dataset/assess_SRR", exist_ok=True)
splited_dir = pathlib.Path(os.environ["DATA_DIR"]) / "splited"
with open("AI/dataset/assess_SRR/criteras.csv") as fd:
    fd.write("accession,srr,peakNum,meanSignalValue,meanPeakWidth\n")
    for accession in os.listdir(splited_dir):
        for srr in os.listdir(splited_dir / accession):
            df = pd.read_csv(
                splited_dir / accession / srr,
                sep="\t",
                names=[
                    "chrom",
                    "chromStart",
                    "chromEnd",
                    "name",
                    "score",
                    "strand",
                    "signalValue",
                    "pValue",
                    "qValue",
                    "peak",
                ],
            )
            peakNum = len(df)
            meanSignalValue = df["signalValue"].mean().item()
            meanPeakWidth = (df["chromEnd"] - df["chromStart"]).mean().item()
            fd.write(f"{accession},{srr},{peakNum},{meanSignalValue},{meanPeakWidth}\n")
