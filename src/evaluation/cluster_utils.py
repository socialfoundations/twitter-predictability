import fire
from pyprojroot import here
from utils import get_subject_data_path
import os
import shutil


def chunk_subjects(N):
    all_subjects = os.listdir(get_subject_data_path())

    def chunks(l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]

    subject_chunks = chunks(all_subjects, N)

    out_dir = here().joinpath("src", "evaluation", "subjects_chunked")
    out_dir.mkdir(exist_ok=True)  # create directory

    if any(out_dir.iterdir()):
        shutil.rmtree(out_dir)  # remove directory and contents
        out_dir.mkdir(exist_ok=True)  # create directory

    for i, subjects in enumerate(subject_chunks):
        with open(out_dir.joinpath(f"chunk_{i}.txt"), "w") as f:
            f.write("\n".join(subjects))


if __name__ == "__main__":
    fire.Fire()
