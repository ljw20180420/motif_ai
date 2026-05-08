#!/usr/bin/env python

import os
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

minimal_unbind_summit_distance = int(sys.argv[1])
validation_ratio = float(sys.argv[2])
test_ratio = float(sys.argv[3])
seed = int(sys.argv[4])


def balance_pos_neg(df: pd.DataFrame) -> pd.DataFrame:
    unbalanced_num = df.shape[0] - 2 * df["bind"].sum()
    df = (
        pd
        .concat(
            [df, df.query("bind").sample(n=unbalanced_num, replace=True)],
        )
        .sample(frac=1.0)
        .reset_index(drop=True)
    )
    return df


train_data_dir = pathlib.Path(os.environ["DATA_DIR"]) / "train_data"
dfs = []
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
        .sample(n=1)
        .reset_index(drop=True)
    )
    dfs.append(df)

df_train, df_valid_test = train_test_split(
    pd.concat(dfs).reset_index(drop=True),
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
