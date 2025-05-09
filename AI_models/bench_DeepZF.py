#!/usr/bin/env python

import os
from tqdm import tqdm

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取参数和日志记录器
from bind_transformer.config import get_config, get_logger

args = get_config(
    [
        "bind_transformer/config_default.ini",
        "bind_transformer/config_custom.ini",
    ]
)
if args.command != "inference":
    raise Exception("command must be inference")
logger = get_logger(args.log_level)

# 读取度量
logger.warning("load metric")
from bind_transformer.metric import hard_metric, select_threshold

# 读取数据
logger.warning("load data")
import subprocess
from io import StringIO
import pandas as pd
import numpy as np
from datasets import load_dataset
from bind_transformer.load_data import train_validation_test_split, data_collector


ds_protein = load_dataset(
    "csv",
    data_files=(args.data_dir / "protein_data.csv").as_posix(),
    column_names=["accession", "protein", "second", "zinc_num"],
)["train"]
ds = load_dataset(
    "csv",
    data_dir=args.data_dir / "DNA_data",
    column_names=["index", "dna", "bind"],
)
ds = train_validation_test_split(ds, args.validation_ratio, args.test_ratio, args.seed)

# 测试DeepZF
logger.warning("test DeepZF")


def bench_DeepZF(ds, shear_DNA_len):
    dfs = []
    for accession in tqdm(ds_protein["accession"]):
        with open(f"{os.environ['DATA_DIR']}/DeepZF/tempfile", "w") as fd:
            for i, example in enumerate(ds):
                if ds_protein["accession"][example["index"]] != accession:
                    continue
                start = (len(example["dna"]) - shear_DNA_len) // 2 + 1
                fd.write(f""">{i}\n{example["dna"][start : start + shear_DNA_len]}\n""")
        output = subprocess.run(
            [
                "fimo",
                "--best-site",
                "--thresh",
                "1",
                "--no-qvalue",
                "--max-strand",
                "--max-stored-scores",
                "99999999",
                f"{os.environ['DATA_DIR']}/DeepZF/motifs/{accession}.meme",
                f"{os.environ['DATA_DIR']}/DeepZF/tempfile",
            ],
            capture_output=True,
        )
        dfs.append(
            pd.read_csv(
                StringIO(output.stdout.decode()),
                sep="\t",
                names=[
                    "index",
                    "start",
                    "end",
                    "motif",
                    "color",
                    "strand",
                    "score",
                    "pValue",
                    "qValue",
                    "peak",
                ],
            ).loc[:, ["index", "pValue"]]
        )
    os.remove(f"{os.environ['DATA_DIR']}/DeepZF/tempfile")
    return pd.concat(dfs).sort_values("index")


subprocess.run("./bench_DeepZF.sh")
shear_DNA_len = 100
train_eval_df = pd.concat(
    [
        bench_DeepZF(ds["train"], shear_DNA_len),
        bench_DeepZF(ds["validation"], shear_DNA_len),
    ]
)
best_thres = select_threshold(
    -np.log10(train_eval_df["pValue"]),
    ds["train"]["bind"] + ds["validation"]["bind"],
    proba=False,
)
test_df = bench_DeepZF(ds["test"], shear_DNA_len)
with open("DeepZF/bench.log", "w") as fd:
    fd.write(
        f"""DeepZF\n{hard_metric(-np.log10(test_df["pValue"]) >= best_thres, ds["test"]["bind"])}\n"""
    )
