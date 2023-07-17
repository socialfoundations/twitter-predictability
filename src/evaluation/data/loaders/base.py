from abc import ABC, abstractmethod
from time import time
from typing import Optional

from datasets import Dataset, DatasetDict
from pymongo.database import Database


class DataLoadingException(Exception):
    def __init__(self, message, subject_id, *args: object) -> None:
        self.subject_id = subject_id
        message = f"Exception occured while processing {subject_id}: " + message
        super().__init__(message, *args)


class BaseSubjectDataLoader(ABC):
    """
    A data loading class for our experiments. At the base of our experiments are subjects (twitter users).
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    @abstractmethod
    def load_data(self) -> DatasetDict:
        pass


class BaseSubjectDataLoaderFromDB(BaseSubjectDataLoader):
    """
    We load the following:
        - user_eval (250 most recent tweets of subject)
        - user_context
        - peer_context
    The evaluation set of tweets is disjoint from the context tweets.
    """

    def __init__(
        self,
        user_id: str,
        db: Database,
        temporally_consistent_context: bool = True,
        num_tweets=250,
    ) -> None:
        super().__init__(user_id)

        self.db = db
        peers_collection = self.db["peers_collection"]
        self.temporally_consistent_context = temporally_consistent_context
        self.num_tweets = num_tweets

        self.peers_list = list(
            peers_collection.find({"mentioned_by.id": self.user_id}, {"_id": 0})
        )

        self.user_eval: Optional[Dataset] = None
        self.user_context: Optional[Dataset] = None
        self.peer_context: Optional[Dataset] = None

    def _dataset_not_empty(self, dataset, name):
        if dataset.num_rows == 0:
            raise DataLoadingException(
                f"Number of {name} tweets is 0.", subject_id=self.user_id
            )
        else:
            return True

    def _dataset_min_tweets(self, dataset, name, minimum=250):
        if dataset.num_rows < minimum:
            raise DataLoadingException(
                f"Number of {name} tweets is less than {minimum}.",
                subject_id=self.user_id,
            )
        else:
            return True

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
        context = sorted_tweets.select(
            indices[-2 * self.num_tweets : -self.num_tweets]
        )  # 250 second-most recent tweets
        eval = sorted_tweets.select(
            indices[-self.num_tweets :]
        )  # 250 most recent tweets

        if self._dataset_not_empty(eval, "eval") and self._dataset_not_empty(
            context, "user_context"
        ):
            return context, eval

    def _oldest_eval_tweet_timestamp(self):
        oldest_tweet = self.user_eval.sort("created_at")[0]
        return oldest_tweet["created_at"]

    def _peer_filter_condition(self, peer_id):
        base_filter = {
            "author_id": peer_id,
            "referenced_tweets.type": {"$ne": "retweeted"},
            "lang": "en",
        }
        if not self.temporally_consistent_context:
            return base_filter
        else:
            before_date = self._oldest_eval_tweet_timestamp()
            base_filter["created_at"] = {"$lt": before_date}
            return base_filter

    def _peer_context_loader(self):
        timelines_collection = self.db["timelines_collection"]

        peer_tweets = []
        for peer in self.peers_list:
            user_tweets = list(
                timelines_collection.find(
                    self._peer_filter_condition(peer["id"]),
                    {"_id": 0},
                ).limit(50)
            )
            peer_tweets.extend(user_tweets)

        dset = Dataset.from_list(peer_tweets)

        if dset.num_rows > self.num_tweets:
            dset = dset.shuffle().select(range(self.num_tweets))

        if self._dataset_not_empty(dset, "peer_context"):
            return dset

    def _load_subject_dataset(self) -> DatasetDict:
        user_tweets = self._load_user_data()
        self.user_context, self.user_eval = self._context_eval_split(user_tweets)

        self.peer_context = self._peer_context_loader()

        user_dataset = DatasetDict(
            {
                "eval": self.user_eval,
                "user_context": self.user_context,
                "peer_context": self.peer_context,
            }
        )

        return user_dataset

    def _validate_splits(self, dataset_dict):
        # check if length of datasets > 0
        for split in dataset_dict.keys():
            # self._dataset_not_empty(dataset_dict[split], name=split)
            self._dataset_min_tweets(
                dataset_dict[split], name=split, minimum=self.num_tweets
            )

    def load_data(self) -> DatasetDict:
        user_dataset = self._load_subject_dataset().sort("created_at", reverse=True)
        self._validate_splits(user_dataset)
        return user_dataset
