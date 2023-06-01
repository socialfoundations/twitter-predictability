from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_prompt_data_path() -> Path:
    return get_project_root().joinpath("out", "prompt_data")


def get_prompt_results_path() -> Path:
    return get_project_root().joinpath("out", "evaluation", "prompt")
