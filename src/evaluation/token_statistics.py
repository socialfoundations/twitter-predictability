import os
import fire
import datasets
import numpy as np
from prompting import load_data, load_tokenizer
from dotenv import load_dotenv
from utils.constants.model import MODEL_TOKENIZER, MODEL_FULLNAME
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from utils import get_prompt_results_path
from collections import Counter
import pickle

load_dotenv()

datasets.logging.set_verbosity_error()

def chars_per_token_model(model="gpt2-xl"):
    tokenizer = load_tokenizer(MODEL_TOKENIZER[model])
    tokens = list(tokenizer.vocab.keys())
    chars = [len(list(tok)) for tok in tokens]
    return np.mean(chars)

def chars_per_token_subject(subject_id, model="gpt2-xl", omit_mentions_hashtags=False):
    data = load_data(mode="multi_control", user_id=subject_id, from_disk=True, remove_mentions_hashtags=omit_mentions_hashtags)
    tokenizer = load_tokenizer(MODEL_TOKENIZER[model])

    all_tweets = data["user_context"]["text"] + data["eval"]["text"]

    chars_per_token = []
    for tweet in tqdm(all_tweets):
        tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_tweet)

        # but this also counts special characters inside the tokens....
        cpt = [len(list(tok)) for tok in tokens]
        chars_per_token.extend(cpt)

    return np.mean(chars_per_token)


def chars_per_token(model="gpt2-xl", omit_mentions_hashtags=False):
    res_path = get_prompt_results_path().joinpath(MODEL_FULLNAME[model])
    assert os.path.exists(res_path)
    subjects = os.listdir(res_path)
    
    avg_chars_per_token = []
    for sub in tqdm(subjects):
        avg_cpt = chars_per_token_subject(sub, model, omit_mentions_hashtags=omit_mentions_hashtags)
        avg_chars_per_token.append(avg_cpt)

    postfix = "_no_mentions_hashtags" if omit_mentions_hashtags else ""
    save_dir = f"token_stats{postfix}/"
    save_dir = Path(save_dir).joinpath(model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    res_file = save_dir.joinpath("cpt.npy")
    with open(res_file, "wb") as f:
        np.save(f, avg_chars_per_token)

def unique_tokens_subject(subject_id, model="gpt2-xl"):
    data = load_data(mode="multi_control", user_id=subject_id, from_disk=True)
    tokenizer = load_tokenizer(MODEL_TOKENIZER[model])

    cnt = Counter()

    all_tweets = data["user_context"]["text"] + data["eval"]["text"]
    
    for tweet in tqdm(all_tweets):
        tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_tweet)

        cnt.update(tokens)

    return len(cnt)


def unique_tokens(model="gpt2-xl", save_dir="token_stats/"):
    res_path = get_prompt_results_path().joinpath(MODEL_FULLNAME[model])
    assert os.path.exists(res_path)
    subjects = os.listdir(res_path)
    
    tokens_per_user = []
    for sub in tqdm(subjects):
        tpu = unique_tokens_subject(sub, model)
        tokens_per_user.append(tpu)

    save_dir = Path(save_dir).joinpath(model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    res_file = save_dir.joinpath("tpu.npy")
    with open(res_file, "wb") as f:
        np.save(f, tokens_per_user)


def per_token_improvement_subject(subject_id, base="none", context="user", model="gpt2-xl"):
    user_res_path = get_prompt_results_path().joinpath(MODEL_FULLNAME[model]).joinpath(subject_id)
    # load base
    base_file = user_res_path.joinpath(f"{base}.npy")
    base_nlls = np.load(base_file)
    # load context
    context_file = user_res_path.joinpath(f"{context}.npy")
    context_nlls = np.load(context_file)
    # calculate diff
    diff = base_nlls - context_nlls
    return diff


def tokens_subject(subject_id, model="gpt2-xl"):
    # load data and tokenizer
    data = load_data(mode="multi_control", user_id=subject_id, from_disk=True)
    tokenizer = load_tokenizer(MODEL_TOKENIZER[model])

    total_tokens = []
    for tweet in data["eval"]["text"]:
        tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_tweet)
        total_tokens.extend(tokens)

    return total_tokens


def token_diff_counter_subject(subject_id, base="none", context="user", model="gpt2-xl"):
    diff = per_token_improvement_subject(subject_id, base, context, model)
    tokens = tokens_subject(subject_id, model)
    token_counter = Counter(tokens)

    token_diff = Counter()
    for d, t in zip(diff, tokens):
        token_diff[t] += d

    return token_diff, token_counter
    

def token_diff_counter(base="none", context="user", model="gpt2-xl", save_dir="token_stats/"):
    save_dir = Path(save_dir).joinpath(model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    res_file1 = save_dir.joinpath(f"token_diff_{base}_{context}.pkl")
    res_file2 = save_dir.joinpath("token_cnt.pkl")

    if not res_file1.exists() or not res_file2.exists():
        # recalculate and save if these don't exist
        res_path = get_prompt_results_path().joinpath(MODEL_FULLNAME[model])
        assert os.path.exists(res_path)
        subjects = os.listdir(res_path)
    
        global_diff, global_cnt = Counter(), Counter()
        for sub in tqdm(subjects):
            diff, cnt = token_diff_counter_subject(sub, base, context, model)
            global_diff += diff
            global_cnt += cnt
            
        with open(res_file1, "wb") as f:
            pickle.dump(global_diff, f)
        with open(res_file2, "wb") as f:
            pickle.dump(global_cnt, f)
    else:
        # load from file
        with open(res_file1, "rb") as f:
            global_diff = pickle.load(f)
        with open(res_file2, "rb") as f:
            global_cnt = pickle.load(f)

    return global_diff, global_cnt
        
if __name__ == '__main__':
  fire.Fire()

