from transformers import Trainer, TrainingArguments
from .model import BindTransformerConfig, BindTransformerModel
from .load_data import data_collector, outputs_train


def train(
    ds,
    train_output_dir,
    seed,
    batch_size,
    optimizer,
    learning_rate,
    scheduler,
    num_epochs,
    warmup_ratio,
    vocab_size,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    intermediate_size,
    hidden_dropout_prob,
    attention_probs_dropout_prob,
    max_position_embeddings,
    pos_weight,
    logger,
):
    """
    For the meanings of parameters, execute: AI_models/run.py -h.
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
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            pos_weight,
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
        data_collator=lambda examples: data_collector(examples, outputs_train),
    )

    logger.info("train model")
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
