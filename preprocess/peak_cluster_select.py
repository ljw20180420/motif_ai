#!/usr/bin/env python

import sys
import os
from signal import signal, SIGPIPE, SIG_IGN
from pathlib import Path
import pandas as pd

signal(SIGPIPE, SIG_IGN)

quantile_value = float(sys.argv[1])

df = pd.read_table(
    sys.stdin,
    header=None,
    names=[
        "chr",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",
        "pValue",
        "qValue",
        "peak",
        "cluster",
    ],
    na_filter=False,
)

# 不用最大值，用quantile，保险一点。
df["pValue_quantile"] = (
    df["pValue"]
    .groupby(df["cluster"])
    .quantile(quantile_value)
    .loc[df["cluster"]]
    .reset_index(drop=True)
)
df["pValue_select"] = (
    df.loc[df["pValue"] >= df["pValue_quantile"]]
    .groupby("cluster")["pValue"]
    .min()
    .loc[df["cluster"]]
    .reset_index(drop=True)
)

try:
    df.loc[
        df["pValue"] == df["pValue_select"],
        [
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
            "cluster",
        ],
    ].to_csv(sys.stdout, sep="\t", header=False, index=False)
except BrokenPipeError:
    pass
