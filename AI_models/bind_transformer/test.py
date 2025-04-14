#!/usr/bin/env python

from torch.utils.data import DataLoader
import shutil
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
import inspect
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import BindTransformerModel
from .pipeline import BindTransformerPipeline
from .load_data import data_collector, outputs_test


def test(
    ds: Dataset,
    proteins: List[Tensor],
    seconds: List[Tensor],
    train_output_dir: Path,
    pipeline_output_dir: Path,
    device: str,
    batch_size: int,
    logger: Logger,
):
    logger.info("load model")
    bind_transformer_model = BindTransformerModel.from_pretrained(train_output_dir).to(
        device
    )
    # remove parent module name
    bind_transformer_model.__module__ = bind_transformer_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = BindTransformerPipeline(bind_transformer_model)

    test_dataloader = DataLoader(
        dataset=ds["test"],
        batch_size=batch_size,
        collate_fn=lambda examples: data_collector(
            examples, proteins, seconds, outputs_test
        ),
    )

    logger.info("test pipeline")
    for batch in test_dataloader:
        output = pipe(batch)

    logger.info("save pipeline")
    pipe.save_pretrained(save_directory=pipeline_output_dir)

    def ignore_func(src, names):
        return [
            name
            for name in names
            if name.startswith(f"{PREFIX_CHECKPOINT_DIR}-") or name.startswith("_")
        ]

    shutil.copyfile("AI_models/bind_transformer/pipeline.py", "pipeline/pipeline.py")

    shutil.copytree(
        train_output_dir,
        pipeline_output_dir
        / inspect.signature(BindTransformerPipeline)[0],  # replace by pipeline module
        ignore=ignore_func,
        dirs_exist_ok=True,
    )
