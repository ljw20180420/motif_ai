#!/usr/bin/env python

import os
import pathlib

import bioframe as bf
import pandas as pd
import py2bit

os.chdir(pathlib.Path(__file__).resolve().parent.parent)


def get_pCBS():
    df = pd.read_csv("summits_256bp.fasta", names=["fa"])
    df_pCBS = pd.DataFrame({
        "chrom": df.loc[0::2, "fa"].str.split("::", expand=True)[1].to_list(),
        "DNA": df.loc[1::2, "fa"].str.upper().to_list(),
    })
    df_pCBS = df_pCBS.assign(
        range=lambda df: df["chrom"].str.split(":", expand=True)[1],
        chrom=lambda df: df["chrom"].str.split(":", expand=True)[0],
        start=lambda df: df["range"].str.split("-", expand=True)[0].astype(int),
        end=lambda df: df["range"].str.split("-", expand=True)[1].astype(int),
    ).drop(columns=["range"])[["chrom", "start", "end", "DNA"]]
    df_pCBS.to_csv("pCBS.csv", index=False)
    df_pCBS[["DNA"]].assign(protein="O88286").to_csv(
        "../../paper/infer_all_models/wiz.csv", index=False
    )


def get_pCBS_train():
    df_pCBS = pd.read_csv("pCBS.csv", header=0)
    wiz_file = (
        pathlib.Path(os.environ["DATA_DIR"]) / "sized" / "O88286.sized.narrowPeak"
    )
    df_wiz = pd.read_csv(
        wiz_file,
        sep="\t",
        names=["chrom", "start", "end", "summit"],
    )
    df_pCBS_train = bf.closest(df_pCBS, df_wiz)
    df_pCBS_train = df_pCBS_train[["chrom_", "start_", "end_"]].rename(
        columns={"chrom_": "chrom", "start_": "start", "end_": "end"}
    )

    with py2bit.open("genome/mm9.2bit") as tb:
        DNAs = []
        for chrom, start, end in zip(
            df_pCBS_train["chrom"], df_pCBS_train["start"], df_pCBS_train["end"]
        ):
            DNA = tb.sequence(chrom, start.item(), end.item())
            DNAs.append(DNA)

    df_pCBS_train["DNA"] = DNAs
    df_pCBS_train.to_csv("pCBS_train.csv", index=False)
    df_pCBS_train[["DNA"]].assign(protein="O88286").to_csv(
        "../../paper/infer_all_models/wiz_train.csv", index=False
    )


get_pCBS()
get_pCBS_train()
