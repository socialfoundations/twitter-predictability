
from pyprojroot import here
from pathlib import Path

base_path = here()


def get_subject_data_path(multi_control=False) -> Path:
    if multi_control:
        return base_path.joinpath("out", "data", "subject_data_m")
    else:
        return base_path.joinpath("out", "data", "subject_data")


def get_prompt_results_path(no_mentions_hashtags=False) -> Path:
    if no_mentions_hashtags:
        return base_path.joinpath("out", "evaluation", "prompt_no_mentions_hashtags")
    else:
        return base_path.joinpath("out", "evaluation", "prompt")

