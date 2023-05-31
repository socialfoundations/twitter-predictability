from dotenv import load_dotenv
from pymongo import MongoClient
import os
import os.path as path
import wandb
from bson.json_util import dumps
import json

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

DATASET_NAME = "tweets-raw"
FILE_PATH = "tweets.json"

filter_options = {
    "filter": {"possibly_sensitive": False},
    "fields": ["id", "author_id", "text"],
}


def filter_collection(collection, max_documents=0, fields=None, filter=None):
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
        run.log_code()

        documents = list(filter_collection(tweets_collection, **filter_options))
        num_docs = len(documents)
        print("Number of documents: %d" % (num_docs))

        metadata = filter_options
        metadata["num_tweets"] = num_docs

        raw_data = wandb.Artifact(
            DATASET_NAME,
            type="dataset",
            description="Tweets that we will use for fine-tuning our LLM.",
            metadata=metadata,
        )

        with raw_data.new_file(FILE_PATH, "w+") as file:
            json_data_list = json.loads(dumps(documents))
            # save data in JSON Line format
            for data in json_data_list:
                json.dump(data, file)
                file.write("\n")

        run.log_artifact(raw_data)
