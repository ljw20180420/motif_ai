#!/usr/bin/env python

import os
import pathlib
import sys

import httpx
import jsonargparse
from common_ai.dataset import MyDatasetAbstract
from huggingface_hub import upload_file

# change to the project folder
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.append(os.getcwd())

from AI.dataset import MyDataset

parser = jsonargparse.ArgumentParser(description="Parse dataset arguments.")
parser.add_argument("--config", action="config")
parser.add_subclass_arguments(baseclass=MyDatasetAbstract, nested_key="dataset")

args = parser.parse_args([
    "--dataset",
    "AI/dataset/dataset.yaml",
    "--dataset.data_dir",
    "AI/dataset/formal_data",
])

ds = MyDataset(**args.dataset.init_args.as_dict())()

while True:
    try:
        ds.push_to_hub("ljw20180420/COP")
        break
    except (RuntimeError, httpx.ConnectError):
        pass

upload_file(
    path_or_fileobj="AI/dataset/formal_data/README.md",
    path_in_repo="README.md",
    repo_id="ljw20180420/COP",
    repo_type="dataset",
)
