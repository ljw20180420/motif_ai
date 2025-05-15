#!/usr/bin/env python

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import shutil
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch.nn.functional as F
import numpy as np
from .model import BindTransformerModel
from .pipeline import BindTransformerPipeline
from .load_data import DataCollator
from .metric import select_threshold, hard_metric
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
    minimal_unbind_summit_distance: int,
    max_num_tokens: int,
    seed: int,
) -> None:
    logger.info("load model")
    bind_transformer_model = BindTransformerModel.from_pretrained(
        train_output_dir / "train"
    ).to(device)
    # remove parent module name
    bind_transformer_model.__module__ = bind_transformer_model.__module__.split(".")[-1]

    logger.info("select threshold")
    data_collator = DataCollator(
        proteins,
        seconds,
        zinc_nums,
        minimal_unbind_summit_distance,
        0.0,  # select negative sample randomly by set select_worst_neg_loss_ratio=0.0
        None,
        dna_length,
        max_num_tokens,
        seed,
    )
    eval_dataloader = DataLoader(
        dataset=ds["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    bind_probabilities, bind = [], []
    for batch in tqdm(eval_dataloader):
        bind_probabilities.append(
            F.sigmoid(
                bind_transformer_model(
                    batch["protein_ids"].to(device),
                    batch["second_ids"].to(device),
                    batch["dna_ids"].to(device),
                )["logit"]
            )
        )
        bind.append(batch["bind"])
    bind_probabilities = torch.cat(bind_probabilities)
    bind = torch.cat(bind)
    best_thres = select_threshold(bind_probabilities, bind)

    logger.info("setup pipeline")
    pipe = BindTransformerPipeline(bind_transformer_model)

    logger.info("test pipeline")
    test_dataloader = DataLoader(
        dataset=ds["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    # test only one batch
    for batch in tqdm(test_dataloader):
        pipe(batch, best_thres)
        break

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

    with open(pipeline_output_dir / "threshold", "w") as fd:
        fd.write(f"{best_thres}\n")
