#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import numpy as np

df = pd.read_table(
    "uniprotkb_ft_zn_fing_C2H2_AND_organism_2025_03_28.tsv",
    sep="\t",
    header=0,
    na_filter=False,
)

df_sec = pd.read_table("secondary_structure.tsv", sep="\t", header=0, na_filter=False)

df = df.merge(df_sec, left_on="Entry", right_on="accession", how="left").fillna("")

# 去掉没有结构的蛋白和长度不符合的蛋白
df = df.loc[df["Length"] == df["sequence"].str.len()].reset_index(drop=True)


def parse_intervals(reg, literals):
    intervalss = []
    for literal in literals:
        intervals = []
        mats = reg.finditer(literal)
        for mat in mats:
            intervals.append([int(mat.group(1)) - 1, int(mat.group(2))])
        intervalss.append(intervals)

    return intervalss


df["disorder"] = parse_intervals(
    re.compile(r'REGION (\d+)\.\.(\d+); /note="Disordered";'), df["Region"]
)
df["zinc_finger"] = parse_intervals(
    re.compile(r'ZN_FING (\d+)\.\.(\d+); /note="C2H2-type \d+";'), df["Zinc finger"]
)
df["KRAB"] = parse_intervals(
    re.compile(r'DOMAIN (\d+)\.\.(\d+); /note="KRAB";'), df["Domain [CC]"]
)

C_re = re.compile(r"[ITSP]")
H_re = re.compile(r"G")
E_re = re.compile(r"B")
secondary_structures = []
for zinc_fingers, KRABs, secondary_structure in zip(
    df["zinc_finger"], df["KRAB"], df["secondary_structure"]
):
    secondary_structure_array = np.array(list(secondary_structure))
    for zinc_finger in zinc_fingers:
        secondary_structure_array[zinc_finger[0] : zinc_finger[1]] = "Z"
    for KRAB in KRABs:
        secondary_structure_array[KRAB[0] : KRAB[1]] = "K"

    new_secondary_structure = "".join(secondary_structure_array)
    new_secondary_structure = C_re.sub("-", new_secondary_structure)
    new_secondary_structure = H_re.sub("H", new_secondary_structure)
    new_secondary_structure = E_re.sub("E", new_secondary_structure)
    secondary_structures.append(new_secondary_structure)

df["secondary_structure"] = secondary_structures


def tokenizer(secondary_structure, step):
    pad_len = int(np.ceil(len(secondary_structure) / step)) * step - len(
        secondary_structure
    )
    secondary_structure += pad_len * "-"
    for i in range(0, len(secondary_structure), step):
        seg = secondary_structure[i : i + step]
        yield max(set(seg), key=seg.count)


df["token"] = [
    "".join(list(tokenizer(secondary_structure, step=10)))
    for secondary_structure in df["secondary_structure"]
]

df.loc[
    :, ["Entry", "Reviewed", "Entry Name", "sequence", "secondary_structure", "token"]
].to_csv("protein.tsv", sep="\t", header=True, index=False)
