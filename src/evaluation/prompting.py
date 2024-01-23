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

    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8 bit precision."},
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "Load model in 4 bit precision. Overrides --load_in_8bit if set."
        },
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

    tweet_by_tweet: bool = field(
        default=False,
        metadata={
            "help": "Don't use sliding window strategy to iterate over eval tweets (like one would with a continuous block of text). Instead, run each tweet through the model separately. Context length will depend on the longest eval tweet."
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

    # more on loading large models: https://huggingface.co/blog/accelerate-large-models
    offload_folder: Optional[str] = field(
        default="offload",
        metadata={
            "help": "Specifies offload folder for model weights. Useful when trying to load a large model that doesn't fit into memory."
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

        if self.tweet_by_tweet:
            if self.window_len or self.stride:
                logger.warning(
                    f"Window length / stride set, however sliding window processing is disabled."
                )

        if self.load_in_8bit and self.load_in_4bit:
            self.load_in_8bit = False
            logger.warning(
                "Both quantization methods (8 bit and 4 bit) were set to true. 4 bit overrides 8 bit, setting --load_in_8bit to False."
            )


def load_data(mode: str, user_id: str, from_disk: bool):
    is_multi_control = mode in ["random_tweet", "random_user", "multi_control"]
    data = (
        load_dataset(
            user_id=user_id,
            from_disk=from_disk,
            data_path=get_subject_data_path(multi_control=is_multi_control),
        )
        .sort("created_at")
        .map(replace_special_characters)
        .map(remove_urls)
        .map(remove_extra_spaces)
    )
    return data


def load_model(
    device: str,
    model_id: str,
    offload_folder: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_safetensors=False,
        device_map="auto",
        offload_folder=offload_folder,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    return model


def load_tokenizer(tokenizer_id: str = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    logger.debug(
        f"Tokenizer\n\t- pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})\n\t- bos token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})\n\t- eos token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})"
    )
    return tokenizer


def _data_model_tokenizer(config: PromptingArguments):

    # load data
    data = load_data(
        mode=config.mode, user_id=config.user_id, from_disk=config.from_disk
    )

    # load model
    model = load_model(
        device=config.device,
        model_id=config.model_id,
        offload_folder=config.offload_folder,
        load_in_8bit=config.load_in_8bit,
    )

    # load tokenizer
    tokenizer_id = (
        config.tokenizer_id if config.tokenizer_id is not None else config.model_id
    )
    tokenizer = load_tokenizer(tokenizer_id=tokenizer_id)

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


def _tokenization_stats(text, tokenizer, name="text"):
    tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
    words = text.split()
    logger.debug(
        f"Total length of {name}: {len(tokens)} tokens ({len(set(tokens))} unique) / {len(words)} words ({len(set(words))} unique)."
    )


def _sliding_window_tokenization(data, tokenizer, window_length, stride, seq_sep, mode):
    tweets = seq_sep.join(data["text"])
    if mode == "none":
        # this ensures we get the probability for generating the first token P(t_1|BOS)
        # even when there is no context preceding the first eval token
        tweets = tokenizer.bos_token + tweets

    _tokenization_stats(
        text=seq_sep.join(data["text"]), tokenizer=tokenizer, name="eval tweets"
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

    return tokenized_tweets


def _tweet_by_tweet_tokenization(data, tokenizer, mode):
    _tokenization_stats(
        text="".join(data["text"]), tokenizer=tokenizer, name="eval tweets"
    )

    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    def add_bos_token(x):
        return {"text": tokenizer.bos_token + x["text"]}

    def tokenize_func(x):
        return tokenizer(
            x["text"], padding=True, return_tensors="pt", add_special_tokens=False
        )

    if mode == "none":
        data = data.map(add_bos_token, keep_in_memory=True)
    tokenized_tweets = data.map(tokenize_func, batched=True, keep_in_memory=True)
    to_keep = ["input_ids", "attention_mask"]
    to_remove = [f for f in tokenized_tweets.features.keys() if f not in to_keep]
    tokenized_tweets = tokenized_tweets.remove_columns(to_remove)
    tokenized_tweets.set_format(type="torch")
    return tokenized_tweets


def _tokenize_eval_data(
    data, tokenizer, window_length, stride, seq_sep, mode, strategy="sliding_window"
):
    assert strategy in ["sliding_window", "tweet_by_tweet"]
    if strategy == "sliding_window":
        logger.info("Tokenizing eval data with a sliding window strategy.")
        tokenized_tweets = _sliding_window_tokenization(
            data, tokenizer, window_length, stride, seq_sep, mode
        )
    elif strategy == "tweet_by_tweet":
        logger.info("Tokenizing eval data with a tweet-by-tweet strategy.")
        tokenized_tweets = _tweet_by_tweet_tokenization(data, tokenizer, mode)
    logger.debug(
        f"Tokenized evaluation data shape (n x window_len): {tokenized_tweets['input_ids'].shape}"
    )
    return tokenized_tweets


def _tokenize_context(tokenizer, context_dataset, context_len, tweet_separator):
    context = tweet_separator.join(context_dataset["text"])

    _tokenization_stats(text=context, tokenizer=tokenizer, name="context tweets")

    tokenizer.truncation_side = (
        "left"  # change to "left" to discard "oldest" context tweets
    )
    tokenized_context = tokenizer(
        context,
        truncation=True,
        max_length=context_len,
        padding=False,
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
        logger.warning(
            f"The provided context (of length {context_length_no_pad}) does not reach specified context length ({context_len})."
        )

    return res


def _tokenized_tweets_context(
    mode, data, tokenizer, window_length, context_length, stride, seq_sep, strategy
):
    tokenized_tweets = _tokenize_eval_data(
        data["eval"],
        tokenizer,
        window_length,
        stride,
        seq_sep=seq_sep,
        mode=mode,
        strategy=strategy,
    )

    if strategy == "tweet_by_tweet":
        l_eval = len(tokenized_tweets["input_ids"][0])
        # TODO: model max length is not set for some tokenizers...
        if tokenizer.model_max_length < 1e6:
            l_max = tokenizer.model_max_length
        else:
            l_max = 4096
        if l_eval > l_max:
            raise ValueError(
                f"Length of longest tokenized eval tweet ({l_eval})) exceeds maximum model sequence length ({l_max})!"
            )
        context_length = l_max - l_eval
        logger.debug(f"Context length set to {context_length}.")

    if mode == "none":
        return tokenized_tweets, None
    if context_length == 0:
        if mode != "none":
            logger.warning(f"Context length is set to 0, but mode is set to {mode}.")
        return tokenized_tweets, None
    else:
        split = mode + "_context"
        if split in data.keys():
            logger.info(f"Tokenizing {split}...")
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


def user_nlls(
    config: PromptingArguments,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
):
    """
    Runs prompting evaluation for all modes.

    Args:
        config (PromptingArguments): Arguments for prompting.
        model (AutoModelForCausalLM): The model used for calculating the negative log likelihoods. Default is None, in which case the model is loaded based on the PromptingArguments.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the eval / context data. Default is None, in which case the tokenizer is loaded based on the PromptingArguments.

    Returns:
        dict | torch.Tensor: If mode is 'all', or 'multi_control' returns a dictionary of tensors.
        In case of 'all':
            The arrays contain token nlls for all of the following modes: ['none', 'user', 'peer', 'random'].
        In case of 'multi_control':
            The arrays contain token nlls for all of the following modes: ['none', 'user', 'peer', 'random_tweet', 'random_user'].
    """
    both_not_set = model is None and tokenizer is None
    both_are_set = model is not None and tokenizer is not None
    assert (
        both_not_set or both_are_set
    ), "Either both model and tokenizer have to be set, or none."
    if model is None and tokenizer is None:
        data, model, tokenizer = _data_model_tokenizer(config)
    else:
        data = load_data(
            mode=config.mode, user_id=config.user_id, from_disk=config.from_disk
        )

    window_length, context_length, stride = None, None, 0
    if config.tweet_by_tweet:
        strategy = "tweet_by_tweet"
    else:
        strategy = "sliding_window"
        window_length, context_length, stride = _window_context_stride(
            config, tokenizer
        )

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
            strategy=strategy,
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
        nll_std = nlls.std(unbiased=True).item()

        print(
            f"Negative log-likelihood (mean +/- ci, std): {nll_mean:.4f} +/- {nll_err:.4f}, {nll_std:.4f}"
        )
        print(
            f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
        )
    elif type(nlls) == dict:
        for mode in nlls.keys():
            print(f"### {mode} mode ###")
            nll_mean, nll_err = torch_compute_confidence_interval(
                nlls[mode], confidence=0.9
            )
            nll_std = nlls[mode].std(unbiased=True).item()

            print(
                f"Negative log-likelihood (mean +/- ci, std): {nll_mean:.4f} +/- {nll_err:.4f}, {nll_std:.4f}"
            )
            print(
                f"Perplexity range: ({np.exp(nll_mean-nll_err):.4f}, {np.exp(nll_mean+nll_err):.4f})"
            )


if __name__ == "__main__":
    main()
