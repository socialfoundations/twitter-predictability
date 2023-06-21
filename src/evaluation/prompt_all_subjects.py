import json
import logging
import os

import numpy as np
from dotenv import load_dotenv
from prompting import (
    PromptingArguments,
    TokenizationError,
    user_nlls,
    all_modes_user_nlls,
)
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import get_prompt_data_path, get_prompt_results_path

load_dotenv()

main_logger = logging.getLogger("main")


def main():
    parser = HfArgumentParser(PromptingArguments)
    parser.add_argument(
        "--subjects_file",
        default=None,
        help="File containing the subject ids that should be processed. If None, then all subjects are procesed that have a dataset.",
    )
    parser.add_argument(
        "--skip_if_exists",
        default=True,
        help="Skip subjects that already have a directory with results.",
    )
    (prompting_args, script_args) = parser.parse_args_into_dataclasses()

    model_name = prompting_args.model_id.split("/")[-1]
    print(f"Running evaluation on {model_name}")

    if script_args.subjects_file is None:
        # get all subjects
        subjects = os.listdir(get_prompt_data_path())
    else:

        if os.path.exists(script_args.subjects_file):
            with open(script_args.subjects_file, "r") as f:
                subjects = f.read().splitlines()
        else:
            raise RuntimeError(f"{script_args.subjects_file} doesn't exist.")

    modes = ["none", "user", "peer", "random"]

    for s_id in tqdm(subjects, desc="subject", position=0):
        if script_args.skip_if_exists:
            user_res_path = (
                get_prompt_results_path().joinpath(model_name).joinpath(s_id)
            )
            path_exists = os.path.exists(user_res_path)
            if path_exists and os.listdir(user_res_path):
                continue
        prompting_args.user_id = s_id
        try:
            # collect nlls for each mode
            results = all_modes_user_nlls(prompting_args)

            # save results
            res_dir = get_prompt_results_path().joinpath(model_name).joinpath(s_id)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            for mode, nlls in results.items():
                res_file = res_dir.joinpath(f"{mode}.npy")
                with open(res_file, "wb") as f:
                    np.save(f, nlls.numpy())
            # save arguments (mode will be the last set mode, eg. 'random')
            json.dump(prompting_args.__dict__, open(res_dir.joinpath("args.json"), "w"))
        except TokenizationError as e:
            main_logger.error(f"Subject id: {s_id}. Error message: {e}")


if __name__ == "__main__":
    main()
