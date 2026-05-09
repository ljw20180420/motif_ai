#!/usr/bin/env python

import os
import pathlib
import sys

import datasets
from common_ai.utils import split_train_valid_test

validation_ratio = float(sys.argv[1])
test_ratio = float(sys.argv[2])
seed = int(sys.argv[3])

balanced_data_dir = pathlib.Path(os.environ["DATA_DIR"]) / "balanced_data"
ds = datasets.load_dataset(path="csv", data_dir=balanced_data_dir)
ds = split_train_valid_test(ds, validation_ratio, test_ratio, seed)

formal_dir = pathlib.Path("formal_data")
os.makedirs(formal_dir, exist_ok=True)
ds["train"].to_csv(formal_dir / "train.csv", index=False)
ds["validation"].to_csv(formal_dir / "validation.csv", index=False)
ds["test"].to_csv(formal_dir / "test.csv", index=False)
