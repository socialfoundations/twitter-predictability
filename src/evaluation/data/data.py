import os

from .dataloaders import (
    SubjectDataLoaderFromDisk,
    PeerAdjustedSubjectDataLoader,
    TemporallyConsistentSubjectDataLoader,
    SubjectDataLoader,
)
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def load_dataset(user_id, from_disk, data_path, strategy="peer_adjusted"):
    if from_disk:
        dl = SubjectDataLoaderFromDisk(user_id=user_id, data_path=data_path)
        return dl.load_data()
    else:
        with MongoClient(os.environ["MONGO_CONN"]) as mongo_conn:
            if strategy == "peer_adjusted":
                dl = PeerAdjustedSubjectDataLoader(
                    user_id=user_id, db=mongo_conn.twitter
                )
            elif strategy == "temporally_consistent":
                dl = TemporallyConsistentSubjectDataLoader(
                    user_id=user_id, db=mongo_conn.twitter
                )
            elif strategy == "none":
                dl = SubjectDataLoader(user_id=user_id, db=mongo_conn.twitter)
            else:
                raise ValueError(
                    "Loading strategy can be one of the following: ['peer_adjusted', 'temporally_consistent', 'none']"
                )
            return dl.load_data()
