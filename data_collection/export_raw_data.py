from dotenv import load_dotenv
from pymongo import MongoClient
import os
import os.path as path
import wandb
from bson.json_util import dumps
import json

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

FILE_PATH = "tweets.json"

filter_options = {
    "filter": {"possibly_sensitive": False},
    "max_documents": 1000000,
    "fields": ["id", "author_id", "text"],
}


def filter_collection(collection, max_documents, fields, filter):
    cursor = collection.find(filter, limit=max_documents, projection=fields)
    return cursor


if __name__ == "__main__":
    # setup MongoDB
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database
    tweets_collection = db.tweets_collection

    with wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity="social-foundations",
        save_code=True,
        job_type="preprocess-data",
    ) as run:
        raw_data = wandb.Artifact(
            "tweets-raw-1M",
            type="dataset",
            description="Tweets that we will use for fine-tuning our LLM.",
            metadata=filter_options,
        )
        cursor = filter_collection(tweets_collection, **filter_options)

        with raw_data.new_file(FILE_PATH, "w+") as file:
            documents = list(cursor)
            json_data = dumps(documents)
            json.dump(json_data, file)

        run.log_artifact(raw_data)
