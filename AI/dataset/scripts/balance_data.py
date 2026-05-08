#!/usr/bin/env python

import os
import pathlib
import sys

import numpy as np
import pandas as pd

minimal_unbind_summit_distance = int(sys.argv[1])
seed = int(sys.argv[2])
rng = np.random.default_rng(seed)

train_data_dir = pathlib.Path(os.environ["DATA_DIR"]) / "train_data"
balanced_data_dir = pathlib.Path(os.environ["DATA_DIR"]) / "balanced_data"
os.makedirs(balanced_data_dir, exist_ok=True)
for file in os.listdir(train_data_dir):
    df = pd.read_csv(train_data_dir / file, header=0).astype({"DNA": "category"})
    value_vars = df.columns.tolist()
    value_vars.remove("protein")
    value_vars.remove("DNA")
    df = (
        df
        .melt(
            id_vars=["protein", "DNA"],
            value_vars=value_vars,
            var_name="actual_protein",
            value_name="distance",
        )
        .assign(bind=lambda df: df["protein"] == df["actual_protein"])
        .query("bind or distance > @minimal_unbind_summit_distance")
        .reset_index(drop=True)
        .drop(columns=["protein", "distance"])
        .rename(columns={"actual_protein": "protein"})
        .groupby(["DNA", "bind"])
        .sample(n=1, random_state=rng)
        .reset_index(drop=True)
    )
    df.to_csv(balanced_data_dir / file, index=False)
