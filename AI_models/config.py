import configargparse
import pathlib
import torch
import logging
import sys


def get_config(config_file=None):
    default_config_files = ["config.ini"]
    if config_file:
        default_config_files = default_config_files + [config_file]

    parser = configargparse.ArgumentParser(
        description="arguments for transcriptor binding DL models",
        default_config_files=default_config_files,
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default="./results", help="output directory"
    )
    parser.add_argument("--seed", type=int, default=63036, help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="WARNING",
        choices=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="set logging level",
    )

    parser_dataset = parser.add_argument_group(
        title="dataset", description="parameters for loading and split dataset"
    )
    parser_dataset.add_argument(
        "--test_ratio", type=float, default=0.1, help="proportion for test samples"
    )
    parser_dataset.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="proportion for validation samples",
    )

    parser_dataloader = parser.add_argument_group(
        title="data loader", description="parameters for data loader"
    )
    parser_dataloader.add_argument(
        "--batch_size", type=int, default=1000, help="batch size"
    )

    parser_optimizer = parser.add_argument_group(
        title="optimizer", description="parameters for optimizer"
    )
    parser_optimizer.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        choices=[
            "adamw_hf",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_apex_fused",
            "adamw_anyprecision",
            "adafactor",
        ],
        help="name of optimizer",
    )
    parser_optimizer.add_argument(
        "--learning_rate", type=float, default=0.001, help="learn rate of the optimizer"
    )

    parser_scheduler = parser.add_argument_group(
        title="scheduler", description="parameters for learning rate scheduler"
    )
    parser_scheduler.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ],
        help="The scheduler type to use.",
    )
    parser_scheduler.add_argument(
        "--num_epochs",
        type=float,
        default=30.0,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser_scheduler.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Ratio of total training steps used for a linear warmup from 0 to learning_rate",
    )

    parser_roformer = parser.add_argument_group(
        title="roformer", description="parameters for roformer"
    )
    parser_roformer.add_argument(
        "--hidden_size", type=int, default=256, help="model embedding dimension"
    )
    parser_roformer.add_argument(
        "--num_hidden_layers", type=int, default=3, help="number of EncoderLayer"
    )
    parser_roformer.add_argument(
        "--num_attention_heads", type=int, default=4, help="number of attention heads"
    )
    parser_roformer.add_argument(
        "--intermediate_size",
        type=int,
        default=1024,
        help="FeedForward intermediate dimension size",
    )
    parser_roformer.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler",
    )
    parser_roformer.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout ratio for the attention probabilities",
    )
    parser_roformer.add_argument(
        "--max_position_embeddings",
        type=int,
        default=32,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).",
    )

    return parser.parse_args()


def get_logger(args):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(args.log)
    logger.addHandler(handler)
    return logger
