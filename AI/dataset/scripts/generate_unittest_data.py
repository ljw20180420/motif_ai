#!/usr/bin/env python

import os
import pathlib

import pandas as pd

os.makedirs("unittest_data", exist_ok=True)

formal_dir = pathlib.Path("formal_data")
unittest_dir = pathlib.Path("unittest_data")
for file, size in zip(["train.csv", "validation.csv", "test.csv"], [4000, 400, 400]):
    pd.read_csv(formal_dir / file, header=0).sample(n=size, random_state=63036).to_csv(
        unittest_dir / file, index=False
    )
