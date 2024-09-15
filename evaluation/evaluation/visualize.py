import fire
import numpy as np
from prompting import load_data, load_tokenizer
from dotenv import load_dotenv
from utils import get_prompt_results_path, to_color, color_text
from sklearn.preprocessing import minmax_scale
from utils.constants.model import MODEL_TOKENIZER, MODEL_FULLNAME
from token_statistics import per_token_improvement_subject

load_dotenv()


def _colorized_text(text_list, normalized_values):
    assert len(text_list) == len(normalized_values), f"Lengths must be equal! {len(text)} != {len(normalized_values)}"
    for val, chunk in zip(normalized_values, text_list):
        rgb = to_color(val)[:-1]
        color_text(chunk, rgb)
    print()


def _plot_eval_tweets_colorized(subject_id, values, model="gpt2-xl", token_level=False, max_tweets=None):
    # normalize values that we are trying to plot
    norm_vals = minmax_scale(values)

    # load data and tokenizer
    data = load_data(mode="multi_control", user_id=subject_id, from_disk=True)
    tokenizer = load_tokenizer(MODEL_TOKENIZER[model])

    start_token = 0
    tweets = data["eval"]["text"] if max_tweets is None else data["eval"]["text"][:max_tweets]
    for tweet in tweets:
        # chunk tweets into individual tokens
        tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_tweet)
        n_tokens = len(tokens)
    
        token_level_vals = norm_vals[start_token:start_token+n_tokens]
        start_token += n_tokens
    
        if token_level:
            # token-level coloring
            _colorized_text(tokens, token_level_vals)
        else:
            # word-level coloring
            word_level_vals = []
            length = 1
            for i in range(n_tokens+1):
                # l = number of words up to i'th token
                l = len(tokenizer.convert_tokens_to_string(tokens[:i]).split())
                if l > length:
                    val = np.max(token_level_vals[length-1:i])
                    # word = tokenizer.convert_tokens_to_string(tokens[:i]).split()[-2]
                    word_level_vals.append(val)
                    # print(f"{len(new_tweet_nlls)}. Word: {word} \t NLL: {nll}")
                    length = l
            # word = tokenizer.convert_tokens_to_string(tokens[:i]).split()[-1]
            last_val = np.max(token_level_vals[length-1:])
            word_level_vals.append(last_val)
            # print(f"{len(new_tweet_nlls)}. Word: {word} \t NLL: {last_nll}")
            words = list(map(lambda word: word + " ", tokenizer.convert_tokens_to_string(tokens).split()))
            _colorized_text(words, word_level_vals)
    

def _plot_legend(values):
    min_, max_ = np.min(values), np.max(values)
    norm_vals = minmax_scale(values)
    min_n_, max_n_ = np.min(norm_vals), np.max(norm_vals)
    color_text("Minimum: " + str(min_), to_color(min_n_)[:-1])
    print()
    color_text("Maximum: " + str(max_), to_color(max_n_)[:-1])
    print()
    color_text(f"Average (token-level): {np.mean(values):.2f}", to_color(np.mean(norm_vals))[:-1])
    print()

def plot_improvement(subject_id, base="none", context="user", model="gpt2-xl", token_level=False, limit=10):
    diff = per_token_improvement_subject(subject_id, base, context, model)
    
    # plot
    _plot_eval_tweets_colorized(subject_id, diff, model, token_level, limit)
    _plot_legend(diff)

def plot_NLLs(subject_id, context="none", model="gpt2-xl", token_level=False, limit=10):
    # load calculated NLLs
    user_res_path = get_prompt_results_path().joinpath(MODEL_FULLNAME[model]).joinpath(subject_id)
    res_file = user_res_path.joinpath(f"{context}.npy")
    nlls = np.load(res_file)

    # plot
    _plot_eval_tweets_colorized(subject_id, nlls, model, token_level, limit)
    _plot_legend(nlls)
    

def _plot_improvement_deprecated(subject_id, context="user", tokenizer_id="gpt2", model_name="gpt2", token_level=False):
    data = load_data(mode="multi_control", user_id=subject_id, from_disk=True)
    tokenizer = load_tokenizer(tokenizer_id)

    # TODO: make sure the right seq sep was used
    seq_sep = "\n"
    eval_text = seq_sep.join(data["eval"]["text"])
    
    token_ids = tokenizer.encode(eval_text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    
    # load calculated nlls
    user_res_path = get_prompt_results_path().joinpath(model_name).joinpath(subject_id)
    res_file = user_res_path.joinpath("none.npy")
    nlls = np.load(res_file)
    res_file = user_res_path.joinpath(f"{context}.npy")
    ctxt_nlls = np.load(res_file)

    # check if the two arrays are of equal length
    assert len(nlls) == len(tokens), f"Length of nlls array and tokenized text must be equal! {len(nlls)} != {len(tokens)}"

    nlls_diff = nlls - ctxt_nlls


    if not token_level:
        new_diff = []
        length = 1
        for i in range(len(tokens)):
            l = len(tokenizer.convert_tokens_to_string(tokens[:i]).split())
            if l > length:
                new_diff.append(np.mean(nlls_diff[length-1:i]))
                length = l
        new_diff.append(np.mean(nlls_diff[length-1:]))
        words = tokenizer.convert_tokens_to_string(tokens).split()
        assert len(words) == len(new_diff), f"Lengths must be equal! {len(words)} != {len(new_diff)}"
        
        norm_diff = minmax_scale(new_diff)
        for diff, word in zip(norm_diff, words):
            rgb = to_color(diff)[:-1]
            color_text(word + " ", rgb)
        print()

        min_, max_ = np.min(new_diff), np.max(new_diff)
        color_text("Minimum: " + str(min_), to_color(min_)[:-1])
        print()
        color_text("Maximum: " + str(max_), to_color(max_)[:-1])
        print()

    else:
        nlls_diff = minmax_scale(nlls_diff)
    
        for diff, token in zip(nlls_diff, tokens):
            rgb = to_color(diff)[:-1]
            color_text(token, rgb)
        print()


if __name__ == '__main__':
  fire.Fire()