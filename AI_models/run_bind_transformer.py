#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from bind_transformer.config import get_config, get_logger
from bind_transformer.load_data import train_validation_test_split

args = get_config(
    [
        "bind_transformer/config_default.ini",
        "bind_transformer/config_custom.ini",
    ]
)
logger = get_logger(args.log_level)

logger.info("loading data")
ds = load_dataset(args.data_dir)
ds = train_validation_test_split(ds, args.validation_ratio, args.test_ratio, args.seed)

if args.command == "train":
    from bind_transformer.train import train

    train(
        ds,
        args.train_output_dir,
        args.seed,
        args.batch_size,
        args.optimizer,
        args.learning_rate,
        args.scheduler,
        args.num_epochs,
        args.warmup_ratio,
        args.vocab_size,
        args.hidden_size,
        args.num_hidden_layers,
        args.num_attention_heads,
        args.intermediate_size,
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.pos_weight,
        logger,
    )
