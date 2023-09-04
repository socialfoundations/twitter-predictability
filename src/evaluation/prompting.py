import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from data import load_dataset
from data.preprocessing import *
from dotenv import load_dotenv
from metrics import negative_log_likelihoods, torch_compute_confidence_interval
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import get_subject_data_path

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("prompting")


@dataclass
class PromptingArguments:
    """
    Arguments for prompting.
    """

    user_id: str = field(
        default="1308026329",
        metadata={"help": "Id of subject that we would like to process."},
    )

    from_disk: bool = field(
        default=False,
        metadata={
            "help": "Whether to load user data from disk or load it from the MongoDB database directly."
        },
    )

    device: Optional[str] = field(
        default="cpu",
        metadata={
            "help": "What device to run experiments on.",
            "choices": ["cpu", "cuda"],
        },
    )

    model_id: str = field(
        default="gpt2", metadata={"help": "The model that we would like evaluate on."}
    )

    tokenizer_id: str = field(
        default=None,
        metadata={
            "help": "The tokenizer we use. If None, then load tokenizer corresponding to model_id."
        },
    )

    ctxt_len: Optional[int] = field(
        default=600,
        metadata={
            "help": "Length of the context that precedes the sequence on which we calculate loss. If None, then it's max_sequence_length - window_length."
        },
    )

    window_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "Length of the sequence on which we calculate our loss. If None, then it's max_sequence_length - context_length."
        },
    )

    stride: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stride strategy when sliding over the evaluation sequence. If None then stride will be half the window length. It is the size of the overlapping chunks."
        },
    )

    mode: str = field(
        default="none",
        metadata={
            "help": "Execution mode. Determines what type of context should be used. If 'all', then it runs all evaluation modes.",
            "choices": [
                "none",
                "user",
                "peer",
                "random",
                "random_tweet",
                "random_user",
                "all",
                "multi_control",
            ],
        },
    )

    seq_sep: str = field(
        default="\n", metadata={"help": "What token(s) to use to separate sequences."}
    )

    batched: bool = field(
        default=False, metadata={"help": "Whether to use batched execution."}
    )

    batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size if batched execution is selected."}
    )

    token_level_nlls: bool = field(
        default=False,
        metadata={
            "help": "Should negative-log-likelihoods be calculated per-token, or per input sequence. The latter gives back an average nll over tokens in that sequence."
        },
    )

    def __post_init__(self):
        if self.ctxt_len == 0 and self.mode != "none":
            logger.warning(
                f"Context length is 0, while mode of running is context={self.mode}. Overriding it to 'none'."
            )
            self.mode = "none"

        if self.seq_sep == "space":
            self.seq_sep = " "
        elif self.seq_sep == "newline":
            self.seq_sep = "\n"

        if self.batch_size is not None and not self.batched:
            logger.warning(
                f"Batch size is set, while --batched is not. Overriding --batched to True."
            )
            self.batched = True


def _data_model_tokenizer(config: PromptingArguments):
    device = torch.device(config.device)

    is_multi_control = config.mode in ["random_tweet", "random_user", "multi_control"]

    # load data
    data = (
        load_dataset(
            user_id=config.user_id,
            from_disk=config.from_disk,
            data_path=get_subject_data_path(multi_control=is_multi_control),
        )
        .sort("created_at")
        .map(replace_special_characters)
        .map(remove_urls)
        .map(remove_extra_spaces)
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, use_safetensors=False
    ).to(device)

    # load tokenizer
    if config.tokenizer_id is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.debug(
        f"Tokenizer\n\t- pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})\n\t- bos token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})\n\t- eos token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})"
    )

    return data, model, tokenizer


def _window_context_stride(config: PromptingArguments, tokenizer):
    if config.window_len is not None and config.ctxt_len is not None:
        # check if total length does not exceed max sequence length
        assert (
            config.window_len + config.ctxt_len <= tokenizer.model_max_length
        ), f"Total length exceeds max sequence length {tokenizer.model_max_length}!"
        window_length, context_length = config.window_len, config.ctxt_len
    elif config.window_len is not None:
        assert (
            config.window_len <= tokenizer.model_max_length
        ), f"Window length exceeds max sequence length {tokenizer.model_max_length}!"
        window_length = config.window_len
        context_length = tokenizer.model_max_length - window_length
    elif config.ctxt_len is not None:
        assert (
            config.ctxt_len <= tokenizer.model_max_length
        ), f"Context length exceeds max sequence length! {tokenizer.model_max_length}"
        context_length = config.ctxt_len
        window_length = tokenizer.model_max_length - context_length
    else:
        raise RuntimeError(
            "Need to specify at least one of these arguments: window_len, ctxt_len"
        )

    if config.stride == None:
        stride = window_length // 2
    else:
        assert config.stride >= 0
        stride = config.stride

    logger.debug(
        f"Window length: {window_length}\nContext length: {context_length}\nStride: {stride}"
    )

    return window_length, context_length, stride


def _tokenize_eval_data(data, tokenizer, window_length, stride, seq_sep, mode):
    tweets = seq_sep.join(data["text"])
    if mode == "none":
        # this ensures we get the probability for generating the first token P(t_1|BOS)
        # even when there is no context preceding the first eval token
        tweets = tokenizer.bos_token + tweets

    tokens_in_eval = len(tokenizer(tweets)["input_ids"])
    logger.debug(
        f"Eval total length: {tokens_in_eval} tokens / {len(tweets.split())} words / {len(data['text'])} tweets."
    )

    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    tokenized_tweets = tokenizer(
        tweets,
        return_overflowing_tokens=True,  # sliding window
        max_length=window_length,
        stride=stride,  # number of overlapping tokens
        truncation=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    logger.debug(
        f"Tokenized evaluation data shape (n x window_len): {tokenized_tweets['input_ids'].shape}"
    )

    return tokenized_tweets


class TokenizationError(RuntimeError):
    pass


def _tokenize_context(tokenizer, context_dataset, context_len, tweet_separator):
    context = tweet_separator.join(context_dataset["text"])

    tokens_in_ctxt = len(tokenizer(context)["input_ids"])
    logger.debug(
        f"Context total length: {tokens_in_ctxt} tokens / {len(context.split())} words / {len(context_dataset['text'])} tweets."
    )

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
        add_special_tokens=False,
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


def _tokenized_tweets_context(
    mode, data, tokenizer, window_length, context_length, stride, seq_sep
):
    tokenized_tweets = _tokenize_eval_data(
        data["eval"],
        tokenizer,
        window_length,
        stride,
        seq_sep=seq_sep,
        mode=mode,
    )
    if mode == "none":
        return tokenized_tweets, None
    else:
        split = mode + "_context"
        if split in data.keys():
            tokenized_context = _tokenize_context(
                tokenizer,
                data[mode + "_context"],
                context_length,
                tweet_separator=seq_sep,
            )
        else:
            raise ValueError(
                f"{split} is not a valid split in the subject's dataset. Available splits: {data.keys()}"
            )
        return tokenized_tweets, tokenized_context


def user_nlls(config: PromptingArguments):
    """
    Runs prompting evaluation for all modes.

    Args:
        config (PromptingArguments): Arguments for prompting.

    Returns:
        dict | torch.Tensor: If mode is 'all', or 'multi_control' returns a dictionary of tensors.
        In case of 'all':
            The arrays contain token nlls for all of the following modes: ['none', 'user', 'peer', 'random'].
        In case of 'multi_control':
            The arrays contain token nlls for all of the following modes: ['none', 'user', 'peer', 'random_tweet', 'random_user'].
    """
    data, model, tokenizer = _data_model_tokenizer(config)

    window_length, context_length, stride = _window_context_stride(config, tokenizer)

    seq_sep_token = torch.tensor(
        tokenizer.encode(config.seq_sep, add_special_tokens=False)
    )
    logger.debug(f"Sequence separator: '{config.seq_sep}' / Encoded: {seq_sep_token}")

    def get_nlls(mode):
        tokenized_tweets, tokenized_context = _tokenized_tweets_context(
            mode,
            data,
            tokenizer=tokenizer,
            window_length=window_length,
            context_length=context_length,
            stride=stride,
            seq_sep=config.seq_sep,
        )

        nlls = negative_log_likelihoods(
            batched=config.batched,
            batch_size=config.batch_size,
            model=model,
            text=tokenized_tweets,
            context=tokenized_context,
            last_ctxt_token=seq_sep_token,
            overlap_len=stride,
            device=torch.device(config.device),
            token_level=config.token_level_nlls,
        )

        nlls = torch.stack(nlls).cpu()
        logger.debug(f"NLLs shape: {nlls.shape}")
        return nlls

    if config.mode == "all":
        results = {}
        for mode in ["none", "user", "peer", "random"]:
            nlls = get_nlls(mode)
            results[mode] = nlls

        return results
    elif config.mode == "multi_control":
        results = {}
        for mode in ["none", "user", "peer", "random_tweet", "random_user"]:
            nlls = get_nlls(mode)
            results[mode] = nlls

        return results
    else:
        return get_nlls(config.mode)


def main():
    parser = HfArgumentParser(PromptingArguments)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, it changes the logging level to debug.",
    )
    (config, args) = parser.parse_args_into_dataclasses()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    nlls = user_nlls(config=config)

    if type(nlls) == torch.Tensor:
        nll_mean, nll_err = torch_compute_confidence_interval(nlls, confidence=0.9)

        print(f"Negative log-likelihood (mean): {nll_mean:.4f} +/- {nll_err:.4f}")
        print(
            f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
        )
    elif type(nlls) == dict:
        for mode in nlls.keys():
            print(f"### {mode} mode ###")
            nll_mean, nll_err = torch_compute_confidence_interval(
                nlls[mode], confidence=0.9
            )

            print(f"Negative log-likelihood (mean): {nll_mean:.4f} +/- {nll_err:.4f}")
            print(
                f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
            )


if __name__ == "__main__":
    main()
