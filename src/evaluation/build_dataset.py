import os, logging

import datasets
from data import load_dataset
from data.loaders import DataLoadingException
from dotenv import load_dotenv
from pymongo import MongoClient
from utils import get_subject_data_path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

load_dotenv()

datasets.disable_progress_bar()

main_logger = logging.getLogger("main")

config = {
    "single_user": False,
    "skip_if_exists": True,
    "multiproc": True,
    "user_id": "1308026329",
    "min_subject_tweets": 500,
}


# save dataset of single user onto disk
def save_single_user_dataset(user_id):
    user_data_path = get_subject_data_path().joinpath(user_id)
    if config["skip_if_exists"]:
        path_exists = os.path.exists(user_data_path)
        if path_exists and os.listdir(user_data_path):
            return

    try:
        user_dataset = load_dataset(
            user_id=user_id, from_disk=False, data_path=get_subject_data_path()
        )
        user_dataset.save_to_disk(user_data_path)
    except DataLoadingException as e:
        main_logger.error(f"Skipping user. {e}")


def main(config):
    with MongoClient(os.environ["MONGO_CONN"]) as mongo_conn:
        db = mongo_conn.twitter  # our database

        if config["single_user"]:
            save_single_user_dataset(db, user_id=config["user_id"])
        else:
            # subjects collection
            subjects_collection = db.subjects_collection

            cursor = subjects_collection.find(
                {
                    "timeline_tweets_count": {"$gte": config["min_subject_tweets"]},
                }
            )

            subject_ids = list(map(lambda x: x["id"], cursor))
            max_ = len(subject_ids)

            if config["multiproc"]:
                count = cpu_count()
                print(f"Starting {count} processes for {max_} subjects...")
                with Pool(count) as p:
                    with tqdm(total=max_) as pbar:
                        for _ in p.imap_unordered(
                            save_single_user_dataset,
                            subject_ids,
                        ):
                            pbar.update(1)
                    pbar.close()
            else:
                for id in tqdm(subject_ids):
                    save_single_user_dataset(id)


if __name__ == "__main__":
    main(config=config)
