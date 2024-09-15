from .base import BaseSubjectDataLoader
from datasets import DatasetDict, load_from_disk
from pathlib import Path


class SubjectDataLoaderFromDisk(BaseSubjectDataLoader):
    def __init__(self, user_id: str, data_path: Path) -> None:
        self.user_data_path = data_path.joinpath(user_id)
        super().__init__(user_id)

    def load_data(self) -> DatasetDict:
        return load_from_disk(self.user_data_path)
