from pyprojroot import here
from pathlib import Path


def get_prompt_data_path() -> Path:
    return here().joinpath("out", "prompt_data")


def get_prompt_results_path() -> Path:
    return here().joinpath("out", "evaluation", "prompt")
