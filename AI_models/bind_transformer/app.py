import gradio as gr
from diffusers import DiffusionPipeline
from .load_data import data_collector, outputs_inference
import torch
import pandas as pd
import torch.nn.functional as F
from ..config import get_config, get_logger

args = get_config(config_file="config_bind_transformer.ini")
logger = get_logger(args)


@torch.no_grad()
def app():
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        "pipeline",
        custom_pipeline="pipeline",
    )
    pipe.bind_transformer_model.to(args.device)

    def gradio_fn(seq):
        batch = data_collector([{"seq": seq}], outputs_inference)

        bind_probability = F.sigmoid(pipe(batch)["logit"].flatten()).item()

        return (bind_probability,)

    gr.Interface(
        fn=gradio_fn,
        inputs=["text"],
        outputs=["number"],
    ).launch()
