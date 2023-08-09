from pyprojroot import here
from pathlib import Path


def get_subject_data_path(multi_control=False) -> Path:
    if multi_control:
        return here().joinpath("out", "subject_data_m")
    else:
        return here().joinpath("out", "subject_data")


def get_prompt_results_path() -> Path:
    return here().joinpath("out", "evaluation", "prompt")
