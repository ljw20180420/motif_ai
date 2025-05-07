import torch
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from .load_data import data_collector
from .tokenizers import (
    DNA_Tokenizer,
    Protein_Bert_Tokenizer,
    Second_Tokenizer,
)


@torch.no_grad()
def inference(
    ds: Dataset,
    proteins: List[Tensor],
    seconds: List[Tensor],
    zinc_nums: List[int],
    pipeline_output_dir: Path,
    device: str,
    logger: Logger,
    batch_size: int,
    dna_length: int,
    max_num_tokens: int,
):
    logger.info("setup data loader")
    inference_dataloader = DataLoader(
        dataset=ds["train"],
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

    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_output_dir, custom_pipeline=pipeline_output_dir.as_posix()
    )
    pipe.bind_transformer_model.to(device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)
