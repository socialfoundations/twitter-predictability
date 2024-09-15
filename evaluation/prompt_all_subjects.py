from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os

import numpy as np
import torch
from prompting import (
    PromptingArguments,
    user_nlls,
    load_model,
    load_tokenizer,
)
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import get_subject_data_path, get_prompt_results_path
from torch.cuda import OutOfMemoryError


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Save results to test/ folder.",
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
        safetensors=prompting_args.safetensors_model,
        local=prompting_args.local_model,
        offload_folder=prompting_args.offload_folder,
        load_in_8bit=prompting_args.load_in_8bit,
        load_in_4bit=prompting_args.load_in_4bit,
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
        model_postfix = ""
        if prompting_args.load_in_8bit:
            model_postfix = "-8bit"
        if prompting_args.load_in_4bit:
            model_postfix =  "-4bit"
            
        if script_args.test:
            res_dir = get_prompt_results_path(prompting_args.omit_mentions_hashtags).joinpath("test" + model_postfix).joinpath(s_id)
        else:
            res_dir = get_prompt_results_path(prompting_args.omit_mentions_hashtags).joinpath(model_name + model_postfix).joinpath(s_id)
        if script_args.skip_if_exists:
            path_exists = os.path.exists(res_dir)
            if path_exists and os.listdir(res_dir):
                continue
        prompting_args.user_id = s_id
        try:
            # collect nlls for each mode
            results = user_nlls(prompting_args, model=model, tokenizer=tokenizer)
            
            # save results
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            if type(results) == dict:
                for mode, nlls in results.items():
                    res_file = res_dir.joinpath(f"{mode}.npy")
                    with open(res_file, "wb") as f:
                        np.save(f, nlls)
            elif type(results) == np.ndarray:
                mode = prompting_args.mode
                res_file = res_dir.joinpath(f"{mode}.npy")
                with open(res_file, "wb") as f:
                    np.save(f, results)
            # save arguments (mode will be the last set mode, eg. 'random')
            json.dump(prompting_args.__dict__, open(res_dir.joinpath("args.json"), "w"))
        except FileNotFoundError as e:
            logger.error(f"Subject data not found ({s_id}). Error message: {e}")
        except OutOfMemoryError as e:
            logger.error(f"Subject id: {s_id}. Error message: {e}")


if __name__ == "__main__":
    main()
