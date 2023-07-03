from .base import BaseSubjectDataLoaderFromDB, DataLoadingException
from datasets import Dataset, DatasetDict
from pymongo.database import Database
from time import time


class SubjectDataLoader(BaseSubjectDataLoaderFromDB):
    def __init__(self, user_id: str, db: Database) -> None:
        super().__init__(user_id, db)

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

    def _dataset_not_empty(self, dataset, name):
        if dataset.num_rows == 0:
            raise DataLoadingException(
                f"Number of {name} tweets is 0.", subject_id=self.user_id
            )
        else:
            return True

    # @timer_func
    def _load_user_data(self):
        timelines_collection = self.db["timelines_collection"]

        user_tweets = list(
            timelines_collection.find(
                {
                    "author_id": self.user_id,
                    "referenced_tweets.type": {"$ne": "retweeted"},
                    "lang": "en",
                },
                {"_id": 0},
            )
        )

        dset = Dataset.from_list(user_tweets)

        if self._dataset_not_empty(dset, name="user"):
            return dset

    def _context_eval_split(self, tweets_dataset: Dataset):
        indices = range(tweets_dataset.num_rows)
        sorted_tweets = tweets_dataset.sort("created_at")  # chronological ordering
        context = sorted_tweets.select(indices[:-250])
        eval = sorted_tweets.select(indices[-250:])  # 250 most recent tweets
        # print(
        #     f"{len(context)} tweets in user context, and {len(eval)} tweets in user eval."
        # )

        if self._dataset_not_empty(eval, "eval") and self._dataset_not_empty(
            context, "user_context"
        ):
            return context, eval

    def _peer_filter_condition(self, peer_id):
        return {
            "author_id": peer_id,
            "referenced_tweets.type": {"$ne": "retweeted"},
            "lang": "en",
        }

    # @timer_func
    def _peer_context_loader(self):
        timelines_collection = self.db["timelines_collection"]

        len_usr_context = len(self.user_context)

        peer_tweets = []
        for peer in self.peers_list:
            user_tweets = list(
                timelines_collection.find(
                    self._peer_filter_condition(peer["id"]),
                    {"_id": 0},
                ).limit(50)
            )
            peer_tweets.extend(user_tweets)

        len_peer_context = len(peer_tweets)

        dset = (
            Dataset.from_list(peer_tweets)
            .sort("created_at", reverse=True)  # newest -> oldest
            .select(range(min(len_usr_context, len_peer_context)))  # select first X
        )

        if self._dataset_not_empty(dset, "peer_context"):
            return dset

    def _random_match_condition(self):
        author_blacklist = [p["id"] for p in self.peers_list]
        author_blacklist.append(self.user_id)

        return {
            "author_id": {"$nin": author_blacklist},
            "referenced_tweets.type": {"$ne": "retweeted"},
            "lang": "en",
        }

    # @timer_func
    def _random_context_loader(self):
        timelines_collection = self.db["timelines_collection"]
        len_usr_context = len(self.user_context)
        rand_user_tweets = list(
            timelines_collection.aggregate(
                [
                    {"$match": self._random_match_condition()},
                    {"$sample": {"size": len_usr_context}},
                    {"$project": {"_id": 0}},
                ]
            )
        )

        dset = Dataset.from_list(rand_user_tweets)

        if self._dataset_not_empty(dset, "random_context"):
            return dset

    def _load_subject_dataset(self) -> DatasetDict:
        user_tweets = self._load_user_data()
        self.user_context, self.user_eval = self._context_eval_split(user_tweets)

        self.peer_context = self._peer_context_loader()

        self.random_context = self._random_context_loader()

        user_dataset = DatasetDict(
            {
                "eval": self.user_eval,
                "user_context": self.user_context,
                "peer_context": self.peer_context,
                "random_context": self.random_context,
            }
        )

        return user_dataset

    def _validate_splits(self, dataset_dict):
        # check if length of datasets > 0
        for split in dataset_dict.keys():
            self._dataset_not_empty(dataset_dict[split], name=split)

    def load_data(self) -> DatasetDict:
        user_dataset = self._load_subject_dataset()
        self._validate_splits(user_dataset)
        return user_dataset


class TemporallyConsistentSubjectDataLoader(SubjectDataLoader):
    """
    A subject data loader where the context tweets happened strictly before the eval tweets.
    """

    def _oldest_eval_tweet_timestamp(self):
        oldest_tweet = self.user_eval.sort("created_at")[0]
        return oldest_tweet["created_at"]

    def _peer_filter_condition(self, peer_id):
        filter_cond = super()._peer_filter_condition(peer_id)
        before_date = self._oldest_eval_tweet_timestamp()
        filter_cond["created_at"] = {"$lt": before_date}
        return filter_cond

    def _random_match_condition(self):
        match_cond = super()._random_match_condition()
        before_date = self._oldest_eval_tweet_timestamp()
        match_cond["created_at"] = {"$lt": before_date}
        return match_cond


class PeerAdjustedSubjectDataLoader(TemporallyConsistentSubjectDataLoader):
    """
    A subject data loader where:
        1) context tweets happened strictly before the eval tweets
        2) subject and random context are matched to peer context timeline-wise

    More on the 2) condition:
        - we select the oldest / newest peer tweets (o_p_t and n_p_t)
        - condition for subject & random context: time(o_p_t) < time(context_tweet) < time(n_p_t)
    """

    def _oldest_peer_tweet_timestamp(self):
        oldest_tweet = self.peer_context.sort("created_at")[0]
        return oldest_tweet["created_at"]

    def _newest_peer_tweet_timestamp(self):
        newest_tweet = self.peer_context.sort("created_at")[-1]
        return newest_tweet["created_at"]

    def _random_match_condition(self):
        match_cond = super()._random_match_condition()
        after_date = self._oldest_peer_tweet_timestamp()
        before_date = self._oldest_eval_tweet_timestamp()
        match_cond["created_at"] = {"$lt": before_date, "$gte": after_date}
        return match_cond

    def _peer_adjusted(self, dataset):
        # filter subject context
        before_date = self._newest_peer_tweet_timestamp()
        after_date = self._oldest_peer_tweet_timestamp()
        # print(f"Filter {after_date} < t < {before_date}")
        res = dataset.filter(
            lambda example: (example["created_at"] < before_date)
            and (example["created_at"] > after_date)
        )
        return res

    def load_data(self) -> DatasetDict:
        user_dataset = self._load_subject_dataset()

        # post-hoc filtering
        user_dataset["user_context"] = self._peer_adjusted(user_dataset["user_context"])
        user_dataset["random_context"] = self._peer_adjusted(
            user_dataset["random_context"]
        )

        self._validate_splits(user_dataset)

        return user_dataset
