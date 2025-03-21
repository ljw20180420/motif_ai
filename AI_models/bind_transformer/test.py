#!/usr/bin/env python

from torch.utils.data import DataLoader
import shutil
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import BindTransformerModel
from .pipeline import BindTransformerPipeline
from .load_data import data_collector, outputs_test


def test(
    ds,
    train_output_dir,
    pipeline_output_dir,
    device,
    batch_size,
    logger,
):
    logger.info("load model")
    bind_transformer_model = BindTransformerModel.from_pretrained(train_output_dir)
    # remove parent module name
    bind_transformer_model.__module__ = bind_transformer_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = BindTransformerPipeline(bind_transformer_model)
    pipe.bind_transformer_model.to(device)

    test_dataloader = DataLoader(
        dataset=ds["test"],
        batch_size=batch_size,
        collate_fn=lambda examples: data_collector(examples, outputs_test),
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
        args.output_dir,
        "pipeline/bind_transformer_model",
        ignore=ignore_func,
        dirs_exist_ok=True,
    )
