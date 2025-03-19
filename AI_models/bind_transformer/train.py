from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import BindTransformerConfig, BindTransformerModel
from ..config import get_config, get_logger
from .load_data import data_collector, outputs_train, train_validation_test_split

args = get_config(config_file="config_bind_transformer.ini")
logger = get_logger(args)


def train(data_files="test/data.csv"):
    logger.info("loading data")
    ds = load_dataset("csv", data_files=data_files)
    ds = train_validation_test_split(ds)

    logger.info("estimate positive weight")
    pos = sum(ds["train"]["bind"])
    neg = ds["train"].num_rows - pos
    pos_weight = neg / pos

    logger.info("initialize model")
    BindTransformerConfig.register_for_auto_class()
    BindTransformerModel.register_for_auto_class()
    bind_transformer_model = BindTransformerModel(
        BindTransformerConfig(
            hidden_size=args.hidden_size,  # model embedding dimension
            num_hidden_layers=args.num_hidden_layers,  # number of EncoderLayer
            num_attention_heads=args.num_attention_heads,  # number of attention heads
            intermediate_size=args.intermediate_size,  # FeedForward intermediate dimension size
            hidden_dropout_prob=args.hidden_dropout_prob,  # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,  # The dropout ratio for the attention probabilities
            max_position_embeddings=args.max_position_embeddings,  # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
            pos_weight=pos_weight,  # weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
        )
    )

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=BindTransformerConfig.label_names,
    )
    training_args.set_dataloader(
        train_batch_size=args.batch_size, eval_batch_size=args.batch_size
    )
    training_args.set_optimizer(name=args.optimizer, learning_rate=args.learning_rate)
    training_args.set_lr_scheduler(
        name=args.scheduler, num_epochs=args.num_epochs, warmup_ratio=args.warmup_ratio
    )
    trainer = Trainer(
        model=bind_transformer_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda examples: data_collector(examples, outputs_train),
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
