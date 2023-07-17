import math
from time import time
from typing import Optional

from datasets import Dataset, DatasetDict
from pymongo.database import Database

from .base import BaseSubjectDataLoaderFromDB, DataLoadingException


class SubjectDataLoader(BaseSubjectDataLoaderFromDB):
    """
    We load the following:
        - user_eval (250 most recent tweets of subject)
        - user_context
        - peer_context
        - random_context (control - random tweets)
    Besides the user and peer context, we also load a random context used for control purposes.
    The tweets in random_context are random tweets from the timelines collection, excluding tweets from the subject and their peers.
    """

    def __init__(
        self, user_id: str, db: Database, temporally_consistent_context: bool = True
    ) -> None:
        super().__init__(user_id, db, temporally_consistent_context)
        self.random_context: Optional[Dataset] = None

    def _random_match_condition(self):
        author_blacklist = [p["id"] for p in self.peers_list]
        author_blacklist.append(self.user_id)

        match_condition = {
            "author_id": {"$nin": author_blacklist},
            "referenced_tweets.type": {"$ne": "retweeted"},
            "lang": "en",
        }

        if not self.temporally_consistent_context:
            return match_condition
        else:
            before_date = self._oldest_eval_tweet_timestamp()
            match_condition["created_at"] = {"$lt": before_date}
            return match_condition

    # @BaseSubjectDataLoaderFromDB.timer_func
    def _random_context_loader(self):
        timelines_collection = self.db["timelines_collection"]
        rand_user_tweets = list(
            timelines_collection.aggregate(
                [
                    {"$match": self._random_match_condition()},
                    {"$sample": {"size": self.num_tweets}},
                    {"$project": {"_id": 0}},
                ]
            )
        )

        dset = Dataset.from_list(rand_user_tweets)

        if self._dataset_not_empty(dset, "random_context"):
            return dset

    def _load_subject_dataset(self) -> DatasetDict:
        user_dataset = super()._load_subject_dataset()

        self.random_context = self._random_context_loader()

        user_dataset["random_context"] = self.random_context

        return user_dataset


class MultiControlSubjectDataLoader(BaseSubjectDataLoaderFromDB):
    """
    We load the following:
        - user_eval (250 most recent tweets of subject)
        - user_context
        - peer_context
        - random_user_context (social control)
        - random_tweet_context (temporal control)
    """

    def __init__(
        self, user_id: str, db: Database, temporally_consistent_context: bool = True
    ) -> None:
        super().__init__(user_id, db, temporally_consistent_context)
        self.random_user_context: Optional[Dataset] = None
        self.random_tweet_context: Optional[Dataset] = None

    def timer_func(func):
        # This function shows the execution time of
        # the function object passed
        def wrap_func(self, *args, **kwargs):
            t1 = time()
            result = func(self, *args, **kwargs)
            t2 = time()
            print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s\r")
            return result

        return wrap_func

    def _rand_user_filter_condition(self, user_id):
        base_filter = {
            "author_id": user_id,
            "referenced_tweets.type": {"$ne": "retweeted"},
            "lang": "en",
        }
        return base_filter

    # @timer_func
    def _random_user_context_loader(self):
        timelines_collection = self.db["timelines_collection"]

        author_blacklist = [p["id"] for p in self.peers_list]
        author_blacklist.append(self.user_id)

        num_peers = len(self.peers_list)
        tweets_per_peer = math.ceil(self.num_tweets / num_peers)

        def match_condition():
            base = {
                "author_id": {"$nin": author_blacklist},
                "referenced_tweets.type": {"$ne": "retweeted"},
                "lang": "en",
            }
            if self.temporally_consistent_context:
                before_date = self._oldest_eval_tweet_timestamp()
                base["created_at"] = {"$lt": before_date}
            return base

        rand_users = timelines_collection.aggregate(
            [
                {
                    "$match": match_condition()
                },  # english, non-rt tweets before oldest eval tweet
                {
                    "$group": {"_id": "$author_id", "count": {"$count": {}}}
                },  # create count grouped by author_id
                {
                    "$match": {"count": {"$gte": tweets_per_peer}}
                },  # match users that have at least this many tweets
                {"$sample": {"size": num_peers}},  # sample randomly
            ]
        )
        rand_users = [user["_id"] for user in rand_users]

        rand_user_tweets = []
        for user in rand_users:
            user_tweets = list(
                timelines_collection.find(
                    self._peer_filter_condition(user),
                    {"_id": 0},
                ).limit(tweets_per_peer)
            )
            rand_user_tweets.extend(user_tweets)

        dset = Dataset.from_list(rand_user_tweets)

        if dset.num_rows > self.num_tweets:
            dset = dset.shuffle().select(range(self.num_tweets))

        if self._dataset_not_empty(dset, "random_user_context"):
            return dset

    # @timer_func
    def _random_tweet_context_loader(self):
        timelines_collection = self.db["timelines_collection"]
        author_blacklist = [p["id"] for p in self.peers_list]
        author_blacklist.append(self.user_id)
        tweet_blacklist = []
        tweets = []
        for tweet in self.peer_context:
            created_at = tweet["created_at"]
            # find another tweet that is closest in time
            tweet_candidate = timelines_collection.find_one(
                {
                    "id": {"$nin": tweet_blacklist},
                    "author_id": {"$nin": author_blacklist},
                    "created_at": {"$lt": created_at},
                },
                {"_id": 0},
                sort=[("created_at", -1)],  # new -> old (first result is returned)
            )
            if tweet_candidate is None:
                raise DataLoadingException(
                    "Could not find matching tweet candidate for random tweet context.",
                    self.user_id,
                )
            tweets.append(tweet_candidate)
            tweet_blacklist.append(tweet_candidate["id"])

        dset = Dataset.from_list(tweets)
        if self._dataset_not_empty(dset, "random_tweet_context"):
            return dset

    def _load_subject_dataset(self) -> DatasetDict:
        user_dataset = super()._load_subject_dataset()

        self.random_user_context = self._random_user_context_loader()
        user_dataset["random_user_context"] = self.random_user_context

        self.random_tweet_context = self._random_tweet_context_loader()
        user_dataset["random_tweet_context"] = self.random_tweet_context

        return user_dataset
