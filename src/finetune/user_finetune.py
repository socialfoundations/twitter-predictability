import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import transformers
from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
from preprocessing import *
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import get_subject_data_path, get_subject_models_path
from my_trainer import NoShuffleTrainer

import wandb

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Name of the base model that we want to fine-tune on subject data."
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    subject_id: str = field(
        metadata={"help": "Id of the subject whose data we want to finetune on."}
    )

    finetune_on: str = field(
        default="user",
        metadata={
            "help": "What type of data to use to fine-tune on. Can be one of the following values: 'user', 'peer', 'random'.",
            "choices": ["user", "peer", "random"],
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    no_shuffle: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to shuffle temporally ordered batches of tweets while training."
        },
    )


@dataclass
class RunArguments:
    change_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Whether to change specified output dir to default output dir that depends on the base model, subject id, what data was used for fine-tuning, etc."
        },
    )


def _load_dataset(data_args: DataArguments):
    subject_data_location = get_subject_data_path().joinpath(data_args.subject_id)
    data = load_from_disk(subject_data_location)

    data = DatasetDict(
        {
            "train": data[data_args.finetune_on + "_context"],
            "validation": data["eval"],
        }
    )

    return data.sort("created_at", reverse=True)  # from oldest to newest


def _load_model_tokenizer(model_args: ModelArguments):
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config
    )

    return tokenizer, model


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, RunArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        run_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        job_type="user finetune",
        tags=[data_args.finetune_on],
    )
    run.log_code()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Change output_dir
    if run_args.change_output_dir:
        logger.warning("Changing output dir...")
        training_args.output_dir = get_subject_models_path().joinpath(
            model_args.model_name_or_path.split("/")[-1],  # name of the base model
            data_args.subject_id,  # subject id
            data_args.finetune_on,  # finetuned on 'user', 'peer' or 'random' tweets
        )
        logger.info(f"New output dir: {training_args.output_dir}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = _load_dataset(data_args)

    def preprocessing_steps(examples):
        return remove_extra_spaces_batch(
            remove_urls_batch(replace_special_characters_batch(examples))
        )

    prepocessed_datasets = raw_datasets.map(
        preprocessing_steps,
        batched=True,
    )

    if training_args.do_train:
        column_names = list(prepocessed_datasets["train"].features)
    else:
        column_names = list(prepocessed_datasets["validation"].features)
    text_column_name = "text"

    tokenizer, model = _load_model_tokenizer(model_args)

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name], padding="longest")
        return output

    tokenized_datasets = prepocessed_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    if data_args.no_shuffle:
        logger.info("Using NoShuffleTrainer...")
        Trainerclass = NoShuffleTrainer
    else:
        logger.info("Using Trainer...")
        Trainerclass = Trainer
    trainer = Trainerclass(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    run.finish()


if __name__ == "__main__":
    main()
