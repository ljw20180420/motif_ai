#!/usr/bin/env python

import os
import pathlib
import subprocess

import pandas as pd

os.chdir(pathlib.Path(__file__).resolve().parent.parent)


class GetUniprorID:
    def __init__(self) -> None:
        self.df = pd.read_csv(
            "AI/dataset/uniprot_C2H2_protein_table.tsv", sep="\t", header=0
        )

    def __call__(self, gene_name: str) -> str:
        id = self.df.loc[
            self.df["Entry Name"].str.contains(gene_name.upper() + "_MOUSE"), "Entry"
        ]
        if len(id) == 1:
            return id.item()
        id = self.df.loc[
            self.df["Protein names"].str.contains(
                r"\b" + gene_name + r"\b", regex=True
            ),
            "Entry",
        ]
        if len(id) == 1:
            return id.item()
        id = self.df.loc[
            self.df["Gene Names"].str.contains(r"\b" + gene_name + r"\b", regex=True),
            "Entry",
        ]
        if len(id) == 1:
            return id.item()
        if gene_name == "Zscan2":
            return "Q07230"

        raise NameError("Not found gene name")


def generate_input():
    proteins = pd.read_excel("paper/office/S6.xlsx", skiprows=0, header=1)
    proteins["protein"] = proteins["C2H2-ZFP gene"].map(GetUniprorID())
    pCBSs = pd.read_csv("AI/dataset/pCBS.csv", header=0)[["DNA"]]

    df_occupy = []
    for protein in proteins["protein"]:
        df_occupy.append(pCBSs.assign(protein=protein))
    pd.concat(df_occupy, ignore_index=True).to_csv("paper/pCBS/occupy.csv", index=False)


def predict_on_pCBS(preprocess: str, model_cls: str):
    output_dir = pathlib.Path(os.environ("OUTPUT_DIR")) / "COP_results"
    checkpoints_path = (
        output_dir
        / "formal"
        / "default"
        / "checkpoints"
        / preprocess
        / model_cls
        / "mouse_C2H2"
        / "default"
    ).as_posix()
    logs_path = (
        output_dir
        / "formal"
        / "default"
        / "logs"
        / preprocess
        / model_cls
        / "mouse_C2H2"
        / "default"
    ).as_posix()
    input_file = "paper/pCBS/occupy.csv"
    os.makedirs("paper/pCBS/output", exist_ok=True)
    output_file = f"paper/pCBS/output/{model_cls}.csv"
    subprocess.run(
        args=[
            "./run.py",
            "infer",
            "--config",
            "AI/infer.yaml",
            "--input",
            input_file,
            "--output",
            output_file,
            "--test.checkpoints_path",
            checkpoints_path,
            "--test.logs_path",
            logs_path,
        ]
    )


for preprocess, model_cls in [
    ("LightGBM", "LightGBM"),
    ("XGBoost", "XGBoost"),
    ("XGBoost", "RandomForest"),
    ("XGBoost", "DecisionTree"),
    ("Scikit", "CategoricalNB"),
    ("Scikit", "SGDClassifier"),
    ("Scikit", "Perceptron"),
    ("Scikit", "PassiveAggressiveClassifier"),
    ("Scikit", "SupportVectorMachine"),
    ("DeepZF", "DeepZF"),
    ("COP", "COP"),
]:
    predict_on_pCBS(preprocess, model_cls)
