#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset, Features, Value

from bind_transformer.config import get_config, get_logger

from bind_transformer.load_data import (
    protein_tokenizer,
    second_tokenizer,
    train_validation_test_split,
)

args = get_config(
    [
        "bind_transformer/config_default.ini",
        "bind_transformer/config_custom.ini",
    ]
)
logger = get_logger(args.log_level)


logger.info("loading data")
ds_protein = load_dataset(
    "csv",
    data_files=(args.data_dir / "protein_data.csv").as_posix(),
    column_names=["accession", "protein", "second"],
)["train"]
proteins = [protein_tokenizer(protein) for protein in ds_protein["protein"]]
seconds = [second_tokenizer(second) for second in ds_protein["second"]]

ds = load_dataset(
    "csv", data_dir=args.data_dir / "DNA_data", column_names=["index", "DNA", "bind"]
)
ds = train_validation_test_split(ds, args.validation_ratio, args.test_ratio, args.seed)

if args.command == "train":
    from bind_transformer.train import train

    train(
        ds,
        proteins,
        seconds,
        args.train_output_dir,
        args.seed,
        args.device,
        logger,
        args.batch_size,
        args.optimizer,
        args.learning_rate,
        args.scheduler,
        args.num_epochs,
        args.warmup_ratio,
        args.protein_animo_acids_vocab_size,
        args.protein_secondary_structure_vocab_size,
        args.protein_coarse_grained_size,
        args.protein_max_position_embeddings,
        args.DNA_vocab_size,
        args.DNA_max_position_embeddings,
        args.embedding_size,
        args.hidden_size,
        args.num_attention_heads,
        args.num_hidden_layers,
        args.chunk_size_feed_forward,
        args.intermediate_size,
        args.hidden_act,
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.initializer_range,
        args.layer_norm_eps,
        args.rotary_value,
        args.pos_weight,
    )
