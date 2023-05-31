import re

from datasets import Dataset


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


def load_context_dataset(db, mode, user_id, before_date):
    timelines_collection = db["timelines_collection"]
    peers_collection = db["peers_collection"]

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
        rand_user_tweets = list(
            timelines_collection.aggregate(
                [
                    {
                        "$match": {
                            "referenced_tweets.type": {"$ne": "retweeted"},
                            "lang": "en",
                            "created_at": {"$lt": before_date},
                        }
                    },
                    {"$sample": {"size": 150}},
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
