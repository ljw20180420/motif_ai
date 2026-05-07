#!/usr/bin/env python

import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

sorted_dir = pathlib.Path(os.environ["DATA_DIR"]) / "sorted"

os.makedirs("assess_peak_width", exist_ok=True)
for file in os.listdir(sorted_dir):
    accession = file.split(".")[0]
    print(accession)
    df = pd.read_csv(sorted_dir / file, sep="\t", usecols=[1, 2], header=None)
    (df[2] - df[1]).plot.hist(bins=100).get_figure().savefig(
        f"assess_peak_width/{accession}.png"
    )
    plt.close("all")
