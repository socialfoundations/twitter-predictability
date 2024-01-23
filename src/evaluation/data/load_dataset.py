import os

from .loaders import (
    SubjectDataLoaderFromDisk,
    SubjectDataLoader,
    BaseSubjectDataLoaderFromDB,
    MultiControlSubjectDataLoader,
)
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def load_dataset(user_id, from_disk, data_path, control="single"):
    if from_disk:
        dl = SubjectDataLoaderFromDisk(user_id=user_id, data_path=data_path)
        return dl.load_data()
    else:
        with MongoClient(os.environ["MONGO_CONN"]) as mongo_conn:
            if control == "single":
                dl = SubjectDataLoader(user_id=user_id, db=mongo_conn.twitter)
            elif control == "multi":
                dl = MultiControlSubjectDataLoader(
                    user_id=user_id, db=mongo_conn.twitter
                )
            elif control == "none":
                dl = BaseSubjectDataLoaderFromDB(user_id=user_id, db=mongo_conn.twitter)
            else:
                raise ValueError(
                    "Loading strategy can be one of the following: ['single', 'multi', 'none']"
                )
            return dl.load_data()
