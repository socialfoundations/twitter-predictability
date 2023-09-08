import json
import logging
import os

import numpy as np
from dotenv import load_dotenv
import torch
from prompting import (
    PromptingArguments,
    TokenizationError,
    user_nlls,
    load_model,
    load_tokenizer,
)
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import get_subject_data_path, get_prompt_results_path
from torch.cuda import OutOfMemoryError

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("prompting")


def main():
    parser = HfArgumentParser(PromptingArguments)
    parser.add_argument(
        "--subjects_file",
        default=None,
        help="File containing the subject ids that should be processed. If None, then all subjects are procesed that have a dataset.",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip subjects that already have a directory with results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, it changes the logging level to debug.",
    )
    (prompting_args, script_args) = parser.parse_args_into_dataclasses()
    if script_args.debug:
        logger.setLevel(logging.DEBUG)

    is_multi_control = prompting_args.mode in [
        "random_tweet",
        "random_user",
        "multi_control",
    ]

    model_name = prompting_args.model_id.split("/")[-1]
    logger.info(f"Running evaluation on {model_name}")

    # preload model once into memory, instead of every time we call user_nlls(...)
    model = load_model(
        device=prompting_args.device,
        model_id=prompting_args.model_id,
        offload_folder=prompting_args.offload_folder,
    )
    tokenizer_id = (
        prompting_args.tokenizer_id
        if prompting_args.tokenizer_id is not None
        else prompting_args.model_id
    )
    tokenizer = load_tokenizer(tokenizer_id=tokenizer_id)

    if script_args.subjects_file is None:
        # get all subjects
        subjects = os.listdir(get_subject_data_path(multi_control=is_multi_control))
    else:

        if os.path.exists(script_args.subjects_file):
            with open(script_args.subjects_file, "r") as f:
                subjects = f.read().splitlines()
        else:
            raise RuntimeError(f"{script_args.subjects_file} doesn't exist.")

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
            results = user_nlls(prompting_args, model=model, tokenizer=tokenizer)

            # save results
            res_dir = get_prompt_results_path().joinpath(model_name).joinpath(s_id)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            if type(results) == dict:
                for mode, nlls in results.items():
                    res_file = res_dir.joinpath(f"{mode}.npy")
                    with open(res_file, "wb") as f:
                        np.save(f, nlls.numpy())
            elif type(results) == torch.Tensor:
                mode = prompting_args.mode
                res_file = res_dir.joinpath(f"{mode}.npy")
                with open(res_file, "wb") as f:
                    np.save(f, results.numpy())
            # save arguments (mode will be the last set mode, eg. 'random')
            json.dump(prompting_args.__dict__, open(res_dir.joinpath("args.json"), "w"))
        except TokenizationError as e:
            logger.error(f"Subject id: {s_id}. Error message: {e}")
        except FileNotFoundError as e:
            logger.error(f"Subject data not found ({s_id}). Error message: {e}")
        except OutOfMemoryError as e:
            logger.error(f"Subject id: {s_id}. Error message: {e}")


if __name__ == "__main__":
    main()
