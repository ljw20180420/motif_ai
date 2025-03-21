import configargparse
import pathlib
import torch
import logging
import sys


def get_config(config_files):
    """
    config_files: Files contain hyper-parameters. The later config files will override the former ones.
    For example, if  config_files=['AI_models/bind_transformer/config_default.ini', 'AI_models/bind_transformer/config_custom.ini'], then settings in config_custom.ini will override settings in config_default.ini. A good practice is to put default settings in config_default.ini (do not modify config_default.ini), and then override default behaviors in config_custom.ini.
    """
    parser = configargparse.ArgumentParser(
        description="Arguments for transcriptor binding roformer model.",
        default_config_files=config_files,
    )

    # command parameters
    parser_command = parser.add_argument_group(
        title="command", description="Command parameters."
    )
    parser_command.add_argument(
        "--command",
        type=str,
        required=True,
        choices=["train", "test", "inference", "app"],
        help="Input directory contains csv files with header protein,DNA,bind",
    )

    # common parameters
    parser_common = parser.add_argument_group(
        title="common", description="Common parameters."
    )
    parser_common.add_argument(
        "--data_dir",
        type=pathlib.Path,
        default="test",
        help="Input directory contains csv files with header protein,DNA,bind",
    )
    parser_common.add_argument(
        "--train_output_dir",
        type=pathlib.Path,
        default="results",
        help="Output directory of training process, which contains model checkpoints for epochs.",
    )
    parser_common.add_argument(
        "--pipeline_output_dir",
        type=pathlib.Path,
        default="pipeline",
        help="Output directory to save pipeline.",
    )
    parser_common.add_argument("--seed", type=int, default=63036, help="random seed")
    parser_common.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda or cpu). If not specified, use cuda if available",
    )
    parser_common.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Set logging level.",
    )

    # dataset parameters
    parser_dataset = parser.add_argument_group(
        title="dataset", description="Parameters for loading and split dataset."
    )
    parser_dataset.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="Proportion for validation samples.",
    )
    parser_dataset.add_argument(
        "--test_ratio", type=float, default=0.1, help="Proportion for test samples."
    )

    # data loader parameters
    parser_data_loader = parser.add_argument_group(
        title="data loader", description="Parameters for data loader."
    )
    parser_data_loader.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size."
    )

    # optimizer parameters
    parser_optimizer = parser.add_argument_group(
        title="optimizer", description="Parameters for optimizer."
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
        help="Optimizer for training.",
    )
    parser_optimizer.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learn rate of training."
    )

    # scheduler parameters
    parser_scheduler = parser.add_argument_group(
        title="scheduler", description="Parameters for learning rate scheduler."
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
        help="The learning rate scheduler to use.",
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

    # roformer parameters
    parser_roformer = parser.add_argument_group(
        title="roformer", description="Parameters for roformer."
    )
    parser_roformer.add_argument(
        "--vocab_size",
        type=int,
        default=24,
        help="The vocabulary size of the model. For protein + DNA, it is 24.",
    )
    parser_roformer.add_argument(
        "--hidden_size", type=int, default=256, help="Model embedding dimension."
    )
    parser_roformer.add_argument(
        "--num_hidden_layers", type=int, default=4, help="Number of EncoderLayer."
    )
    parser_roformer.add_argument(
        "--num_attention_heads", type=int, default=4, help="Number of attention heads."
    )
    parser_roformer.add_argument(
        "--intermediate_size",
        type=int,
        default=1024,
        help="FeedForward intermediate dimension size.",
    )
    parser_roformer.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )
    parser_roformer.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout ratio for the attention probabilities.",
    )
    parser_roformer.add_argument(
        "--max_position_embeddings",
        type=int,
        default=64,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).",
    )
    parser_roformer.add_argument(
        "--pos_weight",
        type=float,
        help="Weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data). If not specified, then pos_weight = neg / pos. pos is the number of positive sample. neg is the number of negative sample.",
    )

    return parser.parse_args()


def get_logger(log_level):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger
