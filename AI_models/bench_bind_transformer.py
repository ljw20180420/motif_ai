#!/usr/bin/env python

import os

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


# 读取数据
logger.warning("load data")
from datasets import load_dataset
from bind_transformer.load_data import DataCollator, train_validation_test_split

ds_protein = load_dataset(
    "csv",
    data_files=(args.data_dir / "protein_data.csv").as_posix(),
    column_names=["accession", "protein", "second", "zinc_num"],
)["train"]

ds = load_dataset("json", data_dir=args.data_dir / "DNA_data")
ds = train_validation_test_split(ds, args.validation_ratio, args.test_ratio, args.seed)
data_collater = DataCollator(
    ds_protein["protein"],
    ds_protein["second"],
    ds_protein["zinc_num"],
    args.minimal_unbind_summit_distance,
    0.0,  # select negative sample randomly by set select_worst_neg_loss_ratio=0.0
    None,
    args.dna_length,
    args.max_num_tokens,
    args.seed,
)
ds = ds["test"].map(
    data_collater.neg_map, batched=True, remove_columns=["rn", "distance"]
)

# 测试集测试
from bind_transformer.inference import inference
from bind_transformer.metric import hard_metric
from datasets import Dataset


def test_set_test(ds: Dataset, ds_protein: Dataset) -> None:
    result = hard_metric(
        torch.cat(
            [
                output
                for output in inference(
                    ds.select_columns(["index", "dna"]),
                    ds_protein["protein"],
                    ds_protein["second"],
                    ds_protein["zinc_num"],
                    args.pipeline_output_dir,
                    args.device,
                    logger,
                    args.batch_size,
                    args.dna_length,
                    args.minimal_unbind_summit_distance,
                    args.max_num_tokens,
                    args.seed,
                )
            ]
        ),
        ds["bind"],
    )

    with open("bind_transformer/bench.log", "a") as fd:
        fd.write(f"""test_set\n{result}\n""")


# 随机序列测试
import numpy as np
import torch


def random_seq_test(ds: Dataset, ds_protein: Dataset) -> None:
    min_length = min([len(protein) for protein in ds_protein["protein"]])
    max_length = max([len(protein) for protein in ds_protein["protein"]])

    random_lengths = np.random.randint(min_length, max_length, len(ds_protein))
    proteins = [
        "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), length))
        for length in random_lengths
    ]
    seconds = [
        "".join(np.random.choice(list("HBEGIPTS-KZ"), length))
        for length in random_lengths
    ]

    result = hard_metric(
        torch.cat(
            [
                output
                for output in inference(
                    ds.select_columns(["index", "dna"]),
                    proteins,
                    seconds,
                    ds_protein["zinc_num"],
                    args.pipeline_output_dir,
                    args.device,
                    logger,
                    args.batch_size,
                    args.dna_length,
                    args.minimal_unbind_summit_distance,
                    args.max_num_tokens,
                    args.seed,
                )
            ]
        ),
        ds["bind"],
    )

    with open("bind_transformer/bench.log", "a") as fd:
        fd.write(f"""random_seq\n{result}\n""")


# 随机排列测试
def random_permute_test(ds: Dataset, ds_protein: Dataset) -> None:
    ds_protein_shuffle = ds_protein.shuffle(seed=63036)
    result = hard_metric(
        torch.cat(
            [
                output
                for output in inference(
                    ds.select_columns(["index", "dna"]),
                    ds_protein_shuffle["protein"],
                    ds_protein_shuffle["second"],
                    ds_protein_shuffle["zinc_num"],
                    args.pipeline_output_dir,
                    args.device,
                    logger,
                    args.batch_size,
                    args.dna_length,
                    args.minimal_unbind_summit_distance,
                    args.max_num_tokens,
                    args.seed,
                )
            ]
        ),
        ds["bind"],
    )

    with open("bind_transformer/bench.log", "a") as fd:
        fd.write(f"""random_permute\n{result}\n""")


# 突变锌指蛋白
import re
import pandas as pd


def mutate_zinc_finger_test(ds: Dataset, ds_protein: Dataset) -> None:
    zf_pattern = re.compile(r"..(C)(?:..|....)(C).{12}(H).{3,5}(H)")
    proteins = []
    for protein in ds_protein["protein"]:
        segs, start = [], 0
        for zf in zf_pattern.finditer(protein):
            segs.append(protein[start : zf.span()[0]])
            mut_zf = (
                protein[zf.regs[0][0] : zf.regs[1][0]]
                + np.random.choice(list("ADEFGHIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[1][1] : zf.regs[2][0]]
                + np.random.choice(list("ADEFGHIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[2][1] : zf.regs[3][0]]
                + np.random.choice(list("ACDEFGIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[3][1] : zf.regs[4][0]]
                + np.random.choice(list("ACDEFGIKLMNPQRSTVWY"), 1).item()
            )
            segs.append(mut_zf)
            start = zf.span()[1]
        segs.append(protein[start:])
        proteins.append("".join(segs))

    df = pd.read_csv("../preprocess/secondary_structure.tsv", sep="\t", header=0)
    seconds = []
    for accession in ds_protein["accession"]:
        seconds.append(
            df.loc[df.loc[:, "accession"] == accession, "secondary_structure"].item()
        )

    result = hard_metric(
        torch.cat(
            [
                output
                for output in inference(
                    ds.select_columns(["index", "dna"]),
                    proteins,
                    seconds,
                    [0] * len(proteins),
                    args.pipeline_output_dir,
                    args.device,
                    logger,
                    args.batch_size,
                    args.dna_length,
                    args.minimal_unbind_summit_distance,
                    args.max_num_tokens,
                    args.seed,
                )
            ]
        ),
        ds["bind"],
    )

    with open("bind_transformer/bench.log", "a") as fd:
        fd.write(f"""mutate_zinc_finger\n{result}\n""")


mutate_zinc_finger_test(ds, ds_protein)
random_seq_test(ds, ds_protein)
random_permute_test(ds, ds_protein)
test_set_test(ds, ds_protein)
