import gradio as gr
from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from logging import Logger
from .load_data import data_collector
from .tokenizers import (
    DNA_Tokenizer,
    Protein_Bert_Tokenizer,
    Second_Tokenizer,
)


@torch.no_grad()
def app(
    pipeline_output_dir: Path,
    device: str,
    logger: Logger,
    dna_length: int,
    max_num_tokens: int,
):
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        pipeline_output_dir,
        custom_pipeline=pipeline_output_dir.as_posix(),
    )
    pipe.bind_transformer_model.to(device)
    with open(pipeline_output_dir / "threshold", "r") as fd:
        threshold = float(fd.readline().strip())

    def gradio_fn(protein, second, DNA):
        batch = data_collector(
            examples=[{"index": 0, "dna": DNA}],
            proteins=[protein],
            seconds=[second],
            zinc_nums=[0],
            DNA_tokenizer=DNA_Tokenizer(dna_length),
            protein_tokenizer=Protein_Bert_Tokenizer(max_num_tokens),
            second_tokenizer=Second_Tokenizer(),
        )

        return pipe(batch, threshold)[0].item()

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "text", "text"],
        outputs=["number"],
    ).launch()
