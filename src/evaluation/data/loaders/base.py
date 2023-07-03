from pymongo.database import Database
from datasets import DatasetDict


class DataLoadingException(Exception):
    def __init__(self, message, subject_id, *args: object) -> None:
        self.subject_id = subject_id
        message = f"Exception occured while processing {subject_id}: " + message
        super().__init__(message, *args)


class BaseSubjectDataLoader:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def load_data(self) -> DatasetDict:
        pass


class BaseSubjectDataLoaderFromDB(BaseSubjectDataLoader):
    def __init__(self, user_id: str, db: Database) -> None:
        super().__init__(user_id)

        self.db = db
        peers_collection = self.db["peers_collection"]

        self.peers_list = list(
            peers_collection.find({"mentioned_by.id": self.user_id}, {"_id": 0})
        )
        # print(f"Subject has {len(self.peers_list)} peers.")

    def load_data(self) -> DatasetDict:
        pass
