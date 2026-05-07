#!/usr/bin/env python

import pandas as pd
import py2bit


def load_cpcdh_ranges():
    df = (
        pd
        .read_csv("paper/infer_all_models/summits_256bp.fasta", names=["region"])
        .loc[0::2]
        .reset_index(drop=True)
    )
    df = (
        df["region"]
        .str.split("::", expand=True)[1]
        .str.split(":", expand=True)
        .rename(columns={0: "chrom", 1: "range"})
    )
    df_se = (
        df["range"]
        .str.split("-", expand=True)
        .astype(int)
        .rename(columns={0: "start", 1: "end"})
    )
    df = df.assign(
        start=df_se["start"],
        end=df_se["end"],
    ).drop(columns=["range"])


breakpoint()
