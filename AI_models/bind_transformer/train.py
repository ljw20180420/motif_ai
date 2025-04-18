import scipy.special
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import List
from torch import Tensor
import scipy
from .model import BindTransformerConfig, BindTransformerModel
from .load_data import data_collector, outputs_train
from .metric import compute_metrics


def train(
    ds: Dataset,
    proteins: List[Tensor],
    seconds: List[Tensor],
    zinc_nums: List[int],
    train_output_dir: Path,
    seed: int,
    device: str,
    logger: Logger,
    batch_size: int,
    DNA_length: int,
    optimizer: str,
    learning_rate: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    protein_animo_acids_vocab_size: int,
    protein_secondary_structure_vocab_size: int,
    protein_coarse_grained_size: int,
    protein_max_position_embeddings: int,
    DNA_vocab_size: int,
    DNA_max_position_embeddings: int,
    embedding_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    chunk_size_feed_forward: int,
    intermediate_size: int,
    hidden_act: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    initializer_range: float,
    layer_norm_eps: float,
    rotary_value: bool,
    pos_weight: float,
):
    """
    For the meanings of parameters, execute: AI_models/run_bind_transformer.py -h.
    """

    logger.info("estimate positive weight")
    if pos_weight is None:
        logger.warning(
            "positive weight is not provided, calculate by negative / positive"
        )
        pos = sum(ds["train"]["bind"])
        neg = ds["train"].num_rows - pos
        pos_weight = neg / pos

    logger.info("initialize model")
    BindTransformerConfig.register_for_auto_class()
    BindTransformerModel.register_for_auto_class()
    bind_transformer_model = BindTransformerModel(
        BindTransformerConfig(
            protein_animo_acids_vocab_size,
            protein_secondary_structure_vocab_size,
            protein_coarse_grained_size,
            protein_max_position_embeddings,
            DNA_vocab_size,
            DNA_max_position_embeddings,
            embedding_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            chunk_size_feed_forward,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            initializer_range,
            layer_norm_eps,
            rotary_value,
            pos_weight,
            seed,
        )
    )

    logger.info("set train arguments")
    training_args = TrainingArguments(
        output_dir=train_output_dir,
        seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=BindTransformerConfig.label_names,
        use_cpu=True if device == "cpu" else False,
        eval_accumulation_steps=1,  # 省点GPU
    )
    training_args.set_dataloader(
        train_batch_size=batch_size, eval_batch_size=batch_size
    )
    training_args.set_optimizer(name=optimizer, learning_rate=learning_rate)
    training_args.set_lr_scheduler(
        name=scheduler, num_epochs=num_epochs, warmup_ratio=warmup_ratio
    )

    trainer = Trainer(
        model=bind_transformer_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda examples: data_collector(
            examples, DNA_length, proteins, seconds, zinc_nums, outputs_train
        ),
        compute_metrics=lambda eval_prediction: compute_metrics(
            eval_prediction.predictions, eval_prediction.label_ids
        ),
    )

    logger.info("train model")
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
