from utils import get_subject_data_path
from datasets import load_from_disk
import os
from random import sample


def get_rand_users(N=1):
    all_subjects = os.listdir(get_subject_data_path())
    return sample(all_subjects, N)

def load_eval(subject_id):
    subject_data_location = get_subject_data_path().joinpath(subject_id)
    data = load_from_disk(subject_data_location)

    return data["eval"]

