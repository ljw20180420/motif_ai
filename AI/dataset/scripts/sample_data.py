#!/usr/bin/env python

import os
import pathlib
import sys

import pandas as pd

sample_num = int(sys.argv[1])
seed = int(sys.argv[2])

data_dir = pathlib.Path(os.environ["DATA_DIR"])
os.makedirs(data_dir / "sampled_data", exist_ok=True)
for file in os.listdir(data_dir / "train_data"):
    df = pd.read_csv(data_dir / "train_data" / file, header=0)
    actual_sample_num = min(sample_num, len(df))
    df = (
        df
        .sample(n=actual_sample_num, random_state=seed)
        .groupby("DNA")
        .sample(n=1, random_state=seed)
    )
    df.to_csv(data_dir / "sampled_data" / file, index=False)
