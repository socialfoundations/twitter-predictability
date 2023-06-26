from pyprojroot import here
from pathlib import Path


def get_subject_data_path() -> Path:
    return here().joinpath("out", "data", "subject_data")


def get_prompt_results_path() -> Path:
    return here().joinpath("out", "evaluation", "prompt")
