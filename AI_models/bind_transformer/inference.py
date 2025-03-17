import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from ..config import get_config, get_logger
from .load_data import data_collector, outputs_inference

args = get_config(config_file="config_bind_transformer.ini")
logger = get_logger(args)


@torch.no_grad()
def inference(data_files="test"):
    logger.info("load inference data")
    ds = load_dataset(
        "csv",
        data_files=data_files,
    )["train"]

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector(examples, outputs_inference),
    )

    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained("pipeline", custom_pipeline="pipeline")
    pipe.bind_transformer_model.to(args.device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)
