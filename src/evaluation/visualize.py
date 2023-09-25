import fire
import numpy as np
from prompting import load_data, load_tokenizer
from dotenv import load_dotenv
from utils import get_prompt_results_path, to_color, color_text
from sklearn.preprocessing import minmax_scale

load_dotenv()
    

def plot_improvement(subject_id, context="user", tokenizer_id="gpt2", model_name="gpt2", token_level=False):
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