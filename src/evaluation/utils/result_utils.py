import os
import fire
import datasets
import numpy as np
from tqdm import tqdm
from functools import wraps
import time
from utils import get_prompt_results_path
from dotenv import load_dotenv
from .constants.model import MODEL_TOKENIZER, MODEL_FULLNAME, MODEL_FULLNAME_FINETUNED
from prompting import load_data, load_tokenizer

load_dotenv()

datasets.logging.set_verbosity_error()

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def subject_tweet_start_indices(subject_id, model="gpt2-xl", from_file=False, finetuned=False, omit_mentions_hashtags=False):
    """
    Returns an array of indices of where each of the subject's tweets start if they were tokenized by a particular model.
    """
    if not from_file:
        data = load_data(mode="multi_control", user_id=subject_id, from_disk=True)
        tokenizer = load_tokenizer(MODEL_TOKENIZER[model])
    
        start_indices = [0]
        for tweet in data["eval"]["text"]:
            tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
            n_tokens = len(tokenized_tweet)
    
            start_indices.append(start_indices[-1]+n_tokens)
    
        # we don't need 0 and the last element
        return start_indices[1:-1]
    else:
        if not finetuned:
            res_path = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME[model]).joinpath(subject_id)
        else:
            res_path = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME_FINETUNED[model]).joinpath(subject_id)
        tsi_file = res_path.joinpath("tweet_start_indices.npy")
        assert tsi_file.exists()
        tsi = np.load(tsi_file)
        return tsi

def tweet_start_indices(model="gpt2-xl", save=True, finetuned=False, omit_mentions_hashtags=False):
    if not finetuned:
        res_path = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME[model])
    else:
        res_path = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME_FINETUNED[model])
    assert os.path.exists(res_path)
    subjects = os.listdir(res_path)

    start_indices = {}
    for s in tqdm(subjects):
        tsi = subject_tweet_start_indices(s, model, finetuned=finetuned, omit_mentions_hashtags=omit_mentions_hashtags)
        start_indices[s] = tsi
        user_res_path = res_path.joinpath(s)
        if save:
            with open(user_res_path.joinpath("tweet_start_indices.npy"), "wb") as f:
                np.save(f, tsi)
            
    return start_indices
        

def subject_nlls(subject_id, mode="none", model="gpt2-xl", finetuned=False, omit_mentions_hashtags=False):
    if not finetuned:
        res_file = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME[model]).joinpath(subject_id).joinpath(f"{mode}.npy")
    else:
        res_file = get_prompt_results_path(omit_mentions_hashtags).joinpath(MODEL_FULLNAME_FINETUNED[model]).joinpath(subject_id).joinpath(f"{mode}.npy")
    assert res_file.exists()

    nlls = np.load(res_file)
    return nlls



if __name__ == "__main__":
    fire.Fire()