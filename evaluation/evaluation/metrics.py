import scipy
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import allocated_memory
from typing import Union
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("prompting")


# From https://discuss.pytorch.org/t/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-classification-and-regression/139398/2
def torch_compute_confidence_interval(
    data: Union[torch.Tensor, np.ndarray], confidence: float = 0.95
) -> torch.Tensor:
    """
    Computes the confidence interval for a given survey of a data set.
    """
    if type(data) == np.ndarray:
        data = torch.from_numpy(data)
    n = len(data)
    mean: torch.Tensor = data.mean()
    # se: Tensor = scipy.stats.sem(data)  # compute standard error
    # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
    se: torch.Tensor = data.std(unbiased=True) / (n**0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1))
    ci = t_p * se
    return mean.item(), ci.item()


def _token_level_nlls(logits, target_ids, device="cpu"):
    # logits dimensions: [bs, input_len, num_tokens]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.to(device)
    # labels dimensions: [bs, input_len]
    shift_labels = target_ids[..., 1:].contiguous()
    shift_labels = shift_labels.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    neg_log_likelihoods = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    # select indices where target_ids wasn't set to -100
    nnz = (shift_labels.flatten() != -100).nonzero()
    return neg_log_likelihoods[nnz].flatten()


def _batched_negative_log_likelihoods(
    model,
    batch_size,
    text,
    context=None,
    last_ctxt_token=None,
    overlap_len=0,
    device="cpu",
    token_level=False,
):
    nlls = []

    if context is not None:
        context_len = len(context["input_ids"])
        num_sequences = text["input_ids"].shape[0]
        # concatenate tensors
        context_ids = context["input_ids"]
        if last_ctxt_token is not None:
            context_ids[context_len - 1] = last_ctxt_token
        input_ids = torch.cat(
            (context_ids.repeat(num_sequences, 1), text["input_ids"]), dim=-1
        )
        attention_mask = torch.cat(
            (
                context["attention_mask"].repeat(num_sequences, 1),
                text["attention_mask"],
            ),
            dim=-1,
        )
        target_ids = torch.where(attention_mask == 0, -100, input_ids)
        # ignore context
        target_ids[0, :context_len] = -100
        if len(target_ids) > 1:
            # ignore context + overlap
            target_ids[1:, : context_len + overlap_len] = -100

    else:
        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]
        target_ids = torch.where(attention_mask == 0, -100, input_ids)
        # ignore overlap
        target_ids[1:, :overlap_len] = -100

    data = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }
    ).with_format("torch")
    batched_data = DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=2)

    pbar = tqdm(batched_data, leave=False)
    for batch in pbar:
        with torch.no_grad():
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["target_ids"].to(device),
            )
            if token_level:
                neg_log_likelihoods = _token_level_nlls(
                    logits=outputs.logits, target_ids=batch["target_ids"], device=device
                ).cpu().detach().numpy()
                nlls.extend(neg_log_likelihoods)
            else:
                neg_log_likelihood = outputs.loss
                nlls.append(neg_log_likelihood)

            pbar.set_description(f"Allocated GPU memory: {allocated_memory():.2f}GB")
            torch.cuda.empty_cache()

    return nlls


def _non_batched_negative_log_likelihoods(
    model,
    text,
    context=None,
    last_ctxt_token=None,
    overlap_len=0,
    device="cpu",
    token_level=False,
):
    first_pass = True
    nlls = []
    for text_input_ids, text_attention_mask in tqdm(
        zip(text["input_ids"], text["attention_mask"]), leave=False
    ):
        if context is not None:
            context_len = len(context["input_ids"])
            # concatenate tensors
            input_ids = torch.cat((context["input_ids"], text_input_ids))
            if last_ctxt_token is not None:
                input_ids[context_len - 1] = last_ctxt_token
            attention_mask = torch.cat((context["attention_mask"], text_attention_mask))
            target_ids = torch.where(attention_mask == 0, -100, input_ids)
            if first_pass:
                # ignore context
                target_ids[:context_len] = -100
            else:
                # ignore context + overlap
                target_ids[: context_len + overlap_len] = -100
        else:
            input_ids = text_input_ids
            attention_mask = text_attention_mask
            target_ids = torch.where(attention_mask == 0, -100, input_ids)
            if not first_pass:
                # ignore overlap
                target_ids[:overlap_len] = -100

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_ids = target_ids.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=target_ids
            )  # the model internally shifts the target ids to the left for next-word prediction
            if token_level:
                neg_log_likelihoods = _token_level_nlls(
                    logits=outputs.logits, target_ids=target_ids, device=device
                ).cpu().detach().numpy()
                nlls.extend(neg_log_likelihoods)
            else:
                neg_log_likelihood = outputs.loss  # the loss is a cross-entropy loss
                nlls.append(neg_log_likelihood)

        first_pass = False

    return nlls


def negative_log_likelihoods(batched, batch_size=2, **kwargs):
    if batched:
        return _batched_negative_log_likelihoods(batch_size=batch_size, **kwargs)
    else:
        return _non_batched_negative_log_likelihoods(**kwargs)


def predictability(hx, z):
    """
    Calculates the upper bound on prediction accuracy (or predictability Π) given Fano's inequality by finding the largest Π that satisfies it.

    Args:
        hx (_type_): Conditional entropy. Uncertainty of our model / hypothesis.
        z (_type_): The cardinality of the text.
    """

    p_list = np.linspace(0, 1, 101)

    predictability = p_list[0]

    def fano(p, h, z):
        # binary entropy with p
        if p == 0 or p == 1:
            h_p = 0
        else:
            h_p = -p * np.log(p) - (1 - p) * np.log(1 - p)

        return h < h_p + (1 - p) * np.log(z - 1)

    for p in p_list:
        if not fano(p, hx, z):
            break
        else:
            predictability = p

    return predictability
