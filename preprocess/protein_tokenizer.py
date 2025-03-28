#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd

df = pd.read_table("protein.tsv", sep="\t", header=0, na_filter=False)

df["token"]
df.columns
df[~df["zinc_finger"]].value_counts("name")
df[~df["zinc_finger"]].value_counts("db_xref")

df.loc[df["name"] == "Helical_region"].value_counts("name")
df.value_counts(["zinc_finger"])
df.loc[~df["zinc_finger"]].value_counts(["coiled_coil"])
df.loc[~df["zinc_finger"]].value_counts(["disordered"])
mask_KRAB = df["name"].str.find("KRAB") != -1
df.loc[
    ~df["zinc_finger"] & ~df["coiled_coil"] & ~df["disordered"] & mask_KRAB
].value_counts("name")
df.loc[
    ~df["zinc_finger"] & ~df["coiled_coil"] & ~df["disordered"] & ~mask_KRAB
].value_counts("db_xref").sum()
