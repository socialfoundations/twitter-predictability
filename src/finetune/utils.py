import pyprojroot
from pathlib import Path

base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))


def get_finetuned_model_path() -> Path:
    return base_path.joinpath("out", "finetune", "base")


def get_subject_models_path() -> Path:
    return base_path.joinpath("out", "finetune", "subject_models")


def get_subject_data_path() -> Path:
    return base_path.joinpath("out", "data", "subject_data")
