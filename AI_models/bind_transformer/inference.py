import torch
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from .load_data import DataCollator
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
    minimal_unbind_summit_distance: int,
    max_num_tokens: int,
    seed: int,
):
    logger.info("setup data loader")
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

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_output_dir, custom_pipeline=pipeline_output_dir.as_posix()
    )
    pipe.bind_transformer_model.to(device)
    with open(pipeline_output_dir / "threshold", "r") as fd:
        threshold = float(fd.readline().strip())

    for batch in tqdm(inference_dataloader):
        yield pipe(batch, threshold)
