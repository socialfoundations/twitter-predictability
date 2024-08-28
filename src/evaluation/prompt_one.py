# load .env variables before loading torch, transformers libraries!!
from dotenv import load_dotenv
load_dotenv()

import os, json
import numpy as np
import logging
import torch
from prompting import (
    PromptingArguments,
    user_nlls
)
from transformers import HfArgumentParser
from utils import get_subject_data_path, get_prompt_results_path, generate_experiment_id
from torch.cuda import OutOfMemoryError


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("prompting")


def main():
    parser = HfArgumentParser(PromptingArguments)
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
        "--name",
        default=None,
        help="Name of the experiment. Results will be put into a directory with that name."
    )


    (prompting_args, script_args) = parser.parse_args_into_dataclasses()
    if script_args.debug:
        logger.setLevel(logging.DEBUG)

    experiment_name = script_args.name
    experiment_id = generate_experiment_id(prompting_args)
    logger.info(f"Running experiment with id={experiment_id}")
    if experiment_name is not None:
        res_dir = get_prompt_results_path(prompting_args.omit_mentions_hashtags).joinpath(experiment_name).joinpath(experiment_id)
    else:
        res_dir = get_prompt_results_path(prompting_args.omit_mentions_hashtags).joinpath("prompt_one").joinpath(experiment_id)
    if script_args.skip_if_exists:
            path_exists = os.path.exists(res_dir)
            if path_exists and os.listdir(res_dir):
                return
    try:
        # collect nlls for each mode
        results = user_nlls(prompting_args)
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
        logger.error(f"{prompting_args}\nSubject data not found. Error message: {e}")
    except OutOfMemoryError as e:
        logger.error(f"{prompting_args}\nError message: {e}")


if __name__ == "__main__":
    main()