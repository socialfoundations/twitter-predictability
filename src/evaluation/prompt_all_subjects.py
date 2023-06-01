from prompting import user_nlls
import os
from dotenv import load_dotenv
from utils import get_prompt_results_path, get_prompt_data_path
from tqdm import tqdm
import numpy as np

load_dotenv()

config = {
    "device": "cuda",
    "model_id": "gpt2",
    "skip_if_exists": True,
}


def main():
    subjects = os.listdir(get_prompt_data_path())

    nlls_config = {
        "from_disk": True,
        "device": config["device"],
        "model_id": config["model_id"],
        "ctxt_len": 900,
        "seq_sep": "\n",
        "batched": True,
        "batch_size": 8,
        "token_level_nlls": True,
    }

    modes = ["none", "user", "peer", "random"]

    for s_id in tqdm(subjects, desc="subject", position=0):
        if config["skip_if_exists"]:
            user_res_path = get_prompt_results_path.join(s_id)
            path_exists = os.path.exists(user_res_path)
            if path_exists and os.listdir(user_res_path):
                continue
        nlls_config["user_id"] = s_id
        for m in tqdm(modes, desc="mode", position=1, leave=False):
            nlls_config["mode"] = m
            nlls = user_nlls(config=nlls_config).cpu().numpy()

            res_dir = get_prompt_results_path().joinpath(s_id)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            res_file = res_dir.joinpath(f"{m}.npy")
            with open(res_file, "wb") as f:
                np.save(f, nlls)


if __name__ == "__main__":
    main()
