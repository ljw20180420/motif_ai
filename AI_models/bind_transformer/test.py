#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import shutil
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import numpy as np
from .model import BindTransformerModel
from .pipeline import BindTransformerPipeline
from .load_data import data_collector
from .metric import compute_metrics_probabilities
from .tokenizers import (
    DNA_Tokenizer,
    Protein_Bert_Tokenizer,
    Second_Tokenizer,
)


@torch.no_grad()
def test(
    ds: Dataset,
    proteins: List[Tensor],
    seconds: List[Tensor],
    zinc_nums: List[int],
    train_output_dir: Path,
    pipeline_output_dir: Path,
    device: str,
    logger: Logger,
    batch_size: int,
    dna_length: int,
    max_num_tokens: int,
):
    logger.info("load model")
    bind_transformer_model = BindTransformerModel.from_pretrained(
        train_output_dir / "train"
    ).to(device)
    # remove parent module name
    bind_transformer_model.__module__ = bind_transformer_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = BindTransformerPipeline(bind_transformer_model)

    test_dataloader = DataLoader(
        dataset=ds["test"],
        batch_size=batch_size,
        collate_fn=lambda examples: data_collector(
            examples,
            proteins,
            seconds,
            zinc_nums,
            DNA_Tokenizer(dna_length),
            Protein_Bert_Tokenizer(max_num_tokens),
            Second_Tokenizer(),
        ),
    )

    logger.info("test pipeline")
    bind_probabilities, binds = [], []
    for batch in test_dataloader:
        bind_probabilities.append(pipe(batch).cpu().numpy())
        binds.append(batch["bind"].cpu().numpy())

    results = compute_metrics_probabilities(
        np.concat(bind_probabilities), np.concat(binds)
    )

    logger.info("save pipeline")
    pipe.save_pretrained(save_directory=pipeline_output_dir)

    def ignore_func(src, names):
        return [
            name
            for name in names
            if name.startswith(f"{PREFIX_CHECKPOINT_DIR}-") or name.startswith("_")
        ]

    shutil.copyfile("bind_transformer/pipeline.py", pipeline_output_dir / "pipeline.py")

    shutil.copytree(
        train_output_dir / "train",
        pipeline_output_dir / list(pipe.components.keys())[0],
        ignore=ignore_func,
        dirs_exist_ok=True,
    )

    return results
