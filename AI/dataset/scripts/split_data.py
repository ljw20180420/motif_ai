#!/usr/bin/env python

import os
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

validation_ratio = float(sys.argv[1])
test_ratio = float(sys.argv[2])
seed = int(sys.argv[3])

balanced_data_dir = pathlib.Path(os.environ["DATA_DIR"]) / "balanced_data"
df = pd.concat([
    pd.read_csv(balanced_data_dir / file, header=0)
    for file in os.listdir(balanced_data_dir)
]).reset_index(drop=True)

df_train, df_valid_test = train_test_split(
    df,
    train_size=1 - validation_ratio - test_ratio,
    random_state=seed,
    shuffle=True,
    stratify=df["bind"],
)
df_valid, df_test = train_test_split(
    df_valid_test,
    train_size=validation_ratio / (validation_ratio + test_ratio),
    random_state=seed,
    shuffle=True,
    stratify=df_valid_test["bind"],
)

balanced_dir = pathlib.Path("balanced")
os.makedirs(balanced_dir, exist_ok=True)
df_train.to_csv(balanced_dir / "train.csv", index=False)
df_valid.to_csv(balanced_dir / "validation.csv", index=False)
df_test.to_csv(balanced_dir / "test.csv", index=False)
