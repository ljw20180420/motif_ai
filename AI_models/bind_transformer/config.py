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
        auto_env_var_prefix="BIND_TRANSFORMER_",
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
        help="What to do.",
    )

    # common parameters
    parser_common = parser.add_argument_group(
        title="common", description="Common parameters."
    )
    parser_common.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Input directory contains csv files with header protein,secondary_structure,DNA,bind",
    )
    parser_common.add_argument(
        "--train_output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory of training process, which contains model checkpoints for epochs.",
    )
    parser_common.add_argument(
        "--pipeline_output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory to save pipeline.",
    )
    parser_common.add_argument("--seed", type=int, required=True, help="random seed")
    parser_common.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda or cpu). If not specified, use cuda if available",
    )
    parser_common.add_argument(
        "--log_level",
        type=str,
        required=True,
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
        required=True,
        help="Proportion for validation samples.",
    )
    parser_dataset.add_argument(
        "--test_ratio", type=float, required=True, help="Proportion for test samples."
    )

    # data loader parameters
    parser_data_loader = parser.add_argument_group(
        title="data loader", description="Parameters for data loader."
    )
    parser_data_loader.add_argument(
        "--batch_size", type=int, required=True, help="Batch size."
    )

    # optimizer parameters
    parser_optimizer = parser.add_argument_group(
        title="optimizer", description="Parameters for optimizer."
    )
    parser_optimizer.add_argument(
        "--optimizer",
        type=str,
        required=True,
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
        "--learning_rate", type=float, required=True, help="Learn rate of training."
    )

    # scheduler parameters
    parser_scheduler = parser.add_argument_group(
        title="scheduler", description="Parameters for learning rate scheduler."
    )
    parser_scheduler.add_argument(
        "--scheduler",
        type=str,
        required=True,
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
        required=True,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser_scheduler.add_argument(
        "--warmup_ratio",
        type=float,
        required=True,
        help="Ratio of total training steps used for a linear warmup from 0 to learning_rate",
    )

    # roformer parameters
    parser_roformer = parser.add_argument_group(
        title="roformer", description="Parameters for roformer."
    )
    parser_roformer.add_argument(
        "--protein_animo_acids_vocab_size",
        type=int,
        required=True,
        help="The vocabulary size of protein animo acids. 20 animo acids and 1 mask token, totally 21.",
    )
    parser_roformer.add_argument(
        "--protein_secondary_structure_vocab_size",
        type=int,
        required=True,
        help="The vocabulary size of protein secondary structure. 11 secondary structrue and 1 mask token, totally 12.",
    )
    parser_roformer.add_argument(
        "--protein_coarse_grained_size",
        type=int,
        required=True,
        help="The coarse-grained bin size of protein.",
    )
    parser_roformer.add_argument(
        "--protein_max_position_embeddings",
        type=int,
        required=True,
        help="The maximum sequence length of protein. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).",
    )
    parser_roformer.add_argument(
        "--DNA_vocab_size",
        type=int,
        required=True,
        help="The vocabulary size of DNA. 4 nucletides and 1 mask token and 1 [CLS] token, totally 6.",
    )
    parser_roformer.add_argument(
        "--DNA_max_position_embeddings",
        type=int,
        required=True,
        help="The maximum sequence length of DNA. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).",
    )
    parser_roformer.add_argument(
        "--embedding_size", type=int, required=True, help="Model embedding dimension."
    )
    parser_roformer.add_argument(
        "--hidden_size", type=int, required=True, help="Model hiddent dimension."
    )
    parser_roformer.add_argument(
        "--num_attention_heads",
        type=int,
        required=True,
        help="Number of attention heads.",
    )
    parser_roformer.add_argument(
        "--num_hidden_layers", type=int, required=True, help="Number of EncoderLayer."
    )
    parser_roformer.add_argument(
        "--chunk_size_feed_forward",
        type=int,
        required=True,
        help="The chunk size of all feed forward layers in the residual attention blocks. A chunk size of 0 means that the feed forward layer is not chunked.",
    )
    parser_roformer.add_argument(
        "--intermediate_size",
        type=int,
        required=True,
        help="FeedForward intermediate dimension size.",
    )
    parser_roformer.add_argument(
        "--hidden_act",
        type=str,
        required=True,
        choices=["gelu", "relu", "selu", "gelu_new"],
        help="The non-linear activation function in the encoder and pooler.",
    )
    parser_roformer.add_argument(
        "--hidden_dropout_prob",
        type=float,
        required=True,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )
    parser_roformer.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        required=True,
        help="The dropout ratio for the attention probabilities.",
    )
    parser_roformer.add_argument(
        "--initializer_range",
        type=float,
        required=True,
        help="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )
    parser_roformer.add_argument(
        "--layer_norm_eps",
        type=float,
        required=True,
        help="The epsilon used by the layer normalization layers.",
    )
    parser_roformer.add_argument(
        "--rotary_value",
        type=bool,
        required=True,
        help="Whether or not apply rotary position embeddings on value layer.",
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
