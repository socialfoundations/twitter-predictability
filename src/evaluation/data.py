import os
import re

from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from pymongo import MongoClient
from utils import get_prompt_data_path

load_dotenv()


def remove_urls(x):
    return {"text": re.sub(r"http\S+", "", x["text"])}


def remove_extra_spaces(x):
    return {"text": " ".join(x["text"].split())}


def load_eval_dataset(db, user_id):
    timelines_collection = db["timelines_collection"]

    user_tweets = list(
        timelines_collection.find(
            {
                "author_id": user_id,
                "referenced_tweets.type": {"$ne": "retweeted"},
                "lang": "en",
            },
            {"_id": 0},
        )
    )

    tweets_dataset = (
        Dataset.from_list(user_tweets)
        .sort("created_at")  # this will give us a chronological ordering
        .select(
            range(len(user_tweets))[-250:]
        )  # we want last 250 tweets (most recent!)
        .map(remove_urls)
        .map(remove_extra_spaces)
    )

    return tweets_dataset


def load_context_dataset(
    db,
    mode,
    user_id,
    before_date,
    after_date=None,
    sample_size=150,
    author_blacklist=[],
):
    timelines_collection = db["timelines_collection"]
    peers_collection = db["peers_collection"]

    assert mode in ["user", "peer", "random"]

    if mode == "user":
        user_tweets = list(
            timelines_collection.find(
                {
                    "author_id": user_id,
                    "referenced_tweets.type": {"$ne": "retweeted"},
                    "lang": "en",
                    "created_at": {"$lt": before_date},
                },
                {"_id": 0},
            )
        )
        context_dataset = (
            Dataset.from_list(user_tweets)
            .sort("created_at")  # this will give us a chronological ordering
            .select(
                range(len(user_tweets))[-250:]
            )  # we want last 250 tweets (most recent!)
            .map(remove_urls)
            .map(remove_extra_spaces)
        )
    elif mode == "peer":
        peers_list = list(
            peers_collection.find({"mentioned_by.id": user_id}, {"_id": 0})
        )
        peer_tweets = []
        for peer in peers_list:
            user_tweets = list(
                timelines_collection.find(
                    {
                        "author_id": peer["id"],
                        "referenced_tweets.type": {"$ne": "retweeted"},
                        "lang": "en",
                        "created_at": {"$lt": before_date},
                    },
                    {"_id": 0},
                ).limit(
                    10
                )  # 10 from each peer
            )
            peer_tweets.extend(user_tweets)
        context_dataset = (
            Dataset.from_list(peer_tweets)
            # .sort("created_at")
            .map(remove_urls)
            .map(remove_extra_spaces)
            .shuffle(56)
        )
    elif mode == "random":
        if after_date is not None:
            time_condition = {"$lt": before_date, "$gte": after_date}
        else:
            time_condition = {"$lt": before_date}
        rand_user_tweets = list(
            timelines_collection.aggregate(
                [
                    {
                        "$match": {
                            "author_id": {"$nin": author_blacklist},
                            "referenced_tweets.type": {"$ne": "retweeted"},
                            "lang": "en",
                            "created_at": time_condition,
                        }
                    },
                    {"$sample": {"size": sample_size}},
                    {"$project": {"_id": 0}},
                ]
            )
        )
        context_dataset = (
            Dataset.from_list(rand_user_tweets)
            .map(remove_urls)
            .map(remove_extra_spaces)
            .shuffle(56)
        )

    return context_dataset


def load_from_database(db, user_id):
    # load data
    tweets_dataset = load_eval_dataset(db, user_id=user_id)

    oldest_tweet = tweets_dataset[0]  # because it's in chronological order
    user_context = load_context_dataset(
        db,
        mode="user",
        user_id=user_id,
        before_date=oldest_tweet["created_at"],
    )
    peer_context = load_context_dataset(
        db,
        mode="peer",
        user_id=user_id,
        before_date=oldest_tweet["created_at"],
    )

    # determine oldest tweet for fair comparison
    num_peer_tweets = len(peer_context)
    # get 90th percentile index
    idx = round(num_peer_tweets * 0.1)
    after_date = peer_context[idx]["created_at"]

    # peers list + subject not to be included in random user authors
    forbidden_authors = set(peer_context[:]["author_id"])
    forbidden_authors.add(user_id)

    random_context = load_context_dataset(
        db,
        mode="random",
        user_id=user_id,
        before_date=oldest_tweet["created_at"],
        after_date=after_date,
        sample_size=num_peer_tweets,
        author_blacklist=list(forbidden_authors),
    )

    user_dataset = DatasetDict(
        {
            "eval": tweets_dataset,
            "user_context": user_context,
            "peer_context": peer_context,
            "random_context": random_context,
        }
    )

    return user_dataset


def load_dataset(user_id, from_disk=True, data_path=get_prompt_data_path()):
    if from_disk:
        return load_from_disk(data_path.joinpath(user_id))
    else:
        mongo_conn = MongoClient(os.environ["MONGO_CONN"])
        db = mongo_conn.twitter  # our database
        return load_from_database(db, user_id=user_id)
