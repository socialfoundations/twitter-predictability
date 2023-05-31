import os

from data import load_from_database
from dotenv import load_dotenv
from pymongo import MongoClient
from utils import get_data_path

load_dotenv()

config = {"user_id": "1308026329"}


# save dataset of single user onto disk
def save_single_user_dataset(db, user_id):
    user_dataset = load_from_database(db, user_id=user_id)

    user_data_path = get_data_path().joinpath(user_id)
    user_dataset.save_to_disk(user_data_path)


def main(config):
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database

    save_single_user_dataset(db, user_id=config["user_id"])


if __name__ == "__main__":
    main(config=config)
