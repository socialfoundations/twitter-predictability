import logging
from dataclasses import dataclass, field

import numpy as np

import torch
from utils import get_other_data_path
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from data.preprocessing import *

from metrics import negative_log_likelihoods, torch_compute_confidence_interval


load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("prompt_other")

@dataclass
class PromptOtherArguments:
    """
    Arguments for prompting other twitter accounts as comparison.
    """

    name: str = field(
        default=None,
        metadata={"help": "Name of the account."}
    )

    model_id: str = field(
        default="gpt2", metadata={"help": "The model that we would like evaluate on."}
    )


def load_data(name: str):
    data_path = get_other_data_path().joinpath(name)
    logger.info(f"Loading data from {data_path}...")
    data = load_dataset("json", data_dir=data_path)
    data = data.rename_column('rawContent', 'text')
    # data = data.map(replace_special_characters).map(remove_urls).map(remove_mentions).map(remove_hashtags).map(remove_extra_spaces)
    data = data.map(replace_special_characters).map(remove_urls).map(remove_extra_spaces)
    return data

def load_model(model_id: str):  
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_safetensors=False,
        device_map="auto"
    )
    return model


def tweet_tokenization(data, tokenizer):
    logger.info("Tokenizing tweets...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    def add_bos_token(x):
        return {"text": tokenizer.bos_token + x["text"]}
    
    def tokenize_func(x):
        return tokenizer(
            x["text"], padding='max_length', return_tensors="pt", add_special_tokens=False, truncation=True, # attention: pads to max_length allowed by model!
        )
    
    data = data.map(add_bos_token, keep_in_memory=True)
    tokenized_tweets = data["train"].map(tokenize_func, batched=True, keep_in_memory=True)
    to_keep = ["input_ids", "attention_mask"]
    to_remove = [f for f in tokenized_tweets.features.keys() if f not in to_keep]
    tokenized_tweets = tokenized_tweets.remove_columns(to_remove)
    tokenized_tweets.set_format(type="torch")
    return tokenized_tweets


def main():
    parser = HfArgumentParser(PromptOtherArguments)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, it changes the logging level to debug.",
    )

    (config, args) = parser.parse_args_into_dataclasses()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # load data
    data = load_data(config.name)

    # load model
    model = load_model(
        model_id=config.model_id,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    # tokenize
    tokenized_tweets = tweet_tokenization(data, tokenizer)

    nlls = negative_log_likelihoods(batched=False, model=model, text=tokenized_tweets, token_level=True)
    # nlls = torch.stack(nlls).cpu()
    nlls = np.stack(nlls)
    logger.debug(f"NLLs shape: {nlls.shape}")

    nll_mean, nll_err = torch_compute_confidence_interval(nlls, confidence=0.9)
    nll_std = nlls.std(unbiased=True).item()
    
    print(
        f"Negative log-likelihood (mean +/- ci, std): {nll_mean:.4f} +/- {nll_err:.4f}, {nll_std:.4f}"
    )
    print(
        f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
    )


if __name__ == "__main__":
    main()


    