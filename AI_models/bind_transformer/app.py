import gradio as gr
from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from logging import Logger
import torch.nn.functional as F
import os
from .load_data import data_collector, outputs_inference


@torch.no_grad()
def app(
    pipeline_output_dir: Path,
    device: str,
    logger: Logger,
):
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_output_dir,
        custom_pipeline=pipeline_output_dir.as_posix(),
    )
    pipe.bind_transformer_model.to(device)

    def gradio_fn(protein, second, DNA):
        batch = data_collector(
            [{"protein": protein, "second": second, "DNA": DNA}], outputs_inference
        )

        bind_probability = F.softmax(pipe(batch), dim=-1)[0, 1].item()

        return (bind_probability,)

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "text", "text"],
        outputs=["number"],
    ).launch()
