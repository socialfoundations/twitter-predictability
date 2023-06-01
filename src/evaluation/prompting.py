import os

import numpy as np
import torch
from data import load_dataset
from dotenv import load_dotenv
from metrics import negative_log_likelihoods, torch_compute_confidence_interval
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

config = {
    "from_disk": True,
    "device": "cpu",
    "user_id": "1308026329",
    "model_id": "gpt2",
    "ctxt_len": 900,
    "mode": "none",
    "seq_sep": "\n",
    "batched": True,
    "batch_size": 2,
    "token_level_nlls": True,
}


class TokenizationError(RuntimeError):
    pass


def tokenize_context(tokenizer, context_dataset, context_len, tweet_separator):
    context = tweet_separator.join(context_dataset["text"])
    tokenizer.truncation_side = (
        "left"  # change to "left" to discard "oldest" context tweets
    )
    tokenizer.padding_side = "left"
    tokenized_context = tokenizer(
        context,
        truncation=True,
        max_length=context_len,
        padding="max_length",  # pad up to max_length
        return_tensors="pt",
    )

    # un-batch result
    # [[context]] -> [context]
    res = {
        "input_ids": tokenized_context["input_ids"].squeeze(),
        "attention_mask": tokenized_context["attention_mask"].squeeze(),
    }

    context_length_no_pad = res["attention_mask"].count_nonzero().item()
    if context_length_no_pad < context_len:
        raise TokenizationError(
            f"The provided context (of length {context_length_no_pad}) does not reach specified context length ({context_len})."
        )

    return res


def user_nlls(config):
    device = torch.device(config["device"])

    # load data
    data = load_dataset(user_id=config["user_id"], from_disk=config["from_disk"])
    tweets_dataset = data["eval"]

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    tweets = config["seq_sep"].join(tweets_dataset["text"])
    window_length = tokenizer.model_max_length - config["ctxt_len"]
    if config["mode"] == "none":
        # this ensures we get the probability for generating the first token P(t_1|BOS)
        # note: there is no difference between eos and bos in gpt2
        tweets = tokenizer.bos_token + tweets
    tokenizer.pad_token = tokenizer.eos_token

    stride = window_length // 2
    tokenized_tweets = tokenizer(
        tweets,
        return_overflowing_tokens=True,  # sliding window
        max_length=window_length,
        stride=stride,  # number of overlapping tokens
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    tokenized_context = None
    if config["mode"] != "none":
        context_dataset = data[config["mode"] + "_context"]

        # tokenize context
        tokenized_context = tokenize_context(
            tokenizer,
            context_dataset,
            context_len=config["ctxt_len"],
            tweet_separator=config["seq_sep"],
        )

    # load model
    model = AutoModelForCausalLM.from_pretrained(config["model_id"]).to(device)

    # calculate nll, perplexity
    sep = torch.tensor(tokenizer.encode(config["seq_sep"]))

    nlls = negative_log_likelihoods(
        batched=config["batched"],
        batch_size=config["batch_size"],
        model=model,
        text=tokenized_tweets,
        context=tokenized_context,
        last_ctxt_token=sep,
        overlap_len=window_length - stride,
        device=device,
        token_level=config["token_level_nlls"],
    )

    return torch.stack(nlls)


def main(config):
    nlls = user_nlls(config=config)

    nll_mean, nll_err = torch_compute_confidence_interval(nlls, confidence=0.9)

    print(f"Negative log-likelihood (mean): {nll_mean:.4f} +/- {nll_err:.4f}")
    print(
        f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
    )


if __name__ == "__main__":
    main(config=config)
