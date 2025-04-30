from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import Union
import optuna
from datetime import datetime
from .tokenizers import (
    DNA_Tokenizer,
    Protein_Bert_Tokenizer,
    Second_Tokenizer,
)
from .model import BindTransformerConfig, BindTransformerModel
from .load_data import data_collector
from .metric import compute_metrics


def train(
    ds: Dataset,
    proteins: list[str],
    seconds: list[str],
    zinc_nums: list[int],
    train_output_dir: Path,
    seed: int,
    device: str,
    fp16: bool,
    logger: Logger,
    batch_size: int,
    dna_length: int,
    optimizer: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    protein_vocab: int,
    second_vocab: int,
    dna_vocab: int,
    max_num_tokens: int,
    dim_emb: int,
    num_heads: int,
    dim_heads: int,
    depth: int,
    dim_ffn: int,
    dropout: float,
    norm_eps: float,
    pos_weight: float,
    reg_l1: float,
    reg_l2: float,
    initializer_range: float,
    hp_study_name: str,
    hp_storage: str,
    redundant_parameters: list[dict[str, Union[str, int, float, list]]],
    n_trials: int,
):
    """
    For the meanings of parameters, execute: AI_models/run_bind_transformer.py -h.
    """

    do_hyperparameter_search = (
        hp_study_name and hp_storage and len(redundant_parameters) and n_trials > 0
    )
    if do_hyperparameter_search:
        hp_study_name = f"""{hp_study_name}.{datetime.now().isoformat()}"""

    logger.info("estimate positive weight")
    if pos_weight == 0.0:
        logger.warning(
            "positive weight is set to 0.0, calculate by negative / positive"
        )
        pos = sum(ds["train"]["bind"])
        neg = ds["train"].num_rows - pos
        pos_weight = neg / pos

    logger.info("initialize model")
    BindTransformerConfig.register_for_auto_class()
    BindTransformerModel.register_for_auto_class()

    def model_init(trial: optuna.trial.Trial) -> BindTransformerModel:
        config = {
            "protein_vocab": protein_vocab,
            "second_vocab": second_vocab,
            "dna_vocab": dna_vocab,
            "max_num_tokens": max_num_tokens,
            "dim_emb": dim_emb,
            "num_heads": num_heads,
            "dim_heads": dim_heads,
            "depth": depth,
            "dim_ffn": dim_ffn,
            "dropout": dropout,
            "norm_eps": norm_eps,
            "pos_weight": pos_weight,
            "reg_l1": reg_l1,
            "reg_l2": reg_l2,
            "initializer_range": initializer_range,
            "seed": seed,
        }

        if trial:
            for key, val in trial.params.items():
                if key in config:
                    config[key] = val
        return BindTransformerModel(BindTransformerConfig(**config))

    logger.info("set train arguments")
    training_args = TrainingArguments(
        output_dir=(
            train_output_dir / "train"
            if not do_hyperparameter_search
            else train_output_dir / hp_study_name
        ),
        eval_strategy="epoch",
        eval_accumulation_steps=1,  # 省GPU
        torch_empty_cache_steps=1,  # 省GPU
        lr_scheduler_kwargs=None,  # 太复杂了
        logging_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True if device == "cpu" else False,
        seed=seed,
        fp16=fp16,
        remove_unused_columns=False,
        label_names=BindTransformerConfig.label_names,
        load_best_model_at_end=True,
    )
    training_args.set_dataloader(
        train_batch_size=batch_size, eval_batch_size=batch_size
    )
    # 不要用优化器自带的weight_decay, 它会对所有参数，包括线性层的偏置和归一化层的参数等都进行L2正则化
    training_args.set_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    training_args.set_lr_scheduler(
        name=scheduler, num_epochs=num_epochs, warmup_ratio=warmup_ratio
    )

    trainer = Trainer(
        args=training_args,
        data_collator=lambda examples: data_collector(
            examples,
            proteins,
            seconds,
            zinc_nums,
            DNA_Tokenizer(dna_length),
            Protein_Bert_Tokenizer(max_num_tokens),
            Second_Tokenizer(),
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        model_init=model_init,
        compute_metrics=lambda eval_prediction: compute_metrics(
            eval_prediction.predictions, eval_prediction.label_ids
        ),
    )

    if not do_hyperparameter_search:
        logger.info("train model")
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError:
            trainer.train()

        logger.info("save model")
        trainer.save_model()
        trainer.create_model_card()
    else:
        logger.info("search hyperparameter")

        def hp_space(trial: optuna.trial.Trial):
            for param in redundant_parameters:
                if param["type"] == "categorical":
                    param.pop("type")
                    trial.suggest_categorical(**param)
                elif param["type"] == "int":
                    param.pop("type")
                    trial.suggest_int(**param)
                else:
                    assert (
                        param["type"] == "float"
                    ), "redundant parameter type is not in categorical, int, float"
                    param.pop("type")
                    trial.suggest_float(**param)
            return trial.params

        def compute_objective(metrics):
            return metrics["eval_loss"]

        # Trainer会自动调用_report_to_hp_search方法，从而调用optuna.TrialPruned()
        trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=compute_objective,
            n_trials=n_trials,
            direction="minimize",
            backend="optuna",
            hp_name=None,
            storage=hp_storage,
            sampler=None,  # 默认optuna.samplers.TPESampler()
            pruner=None,  # 默认optuna.pruners.MedianPruner()
            study_name=hp_study_name,
        )
