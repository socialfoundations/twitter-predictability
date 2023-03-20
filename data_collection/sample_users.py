import os, logging, time
from datetime import datetime
import utils.logging
import utils.tweepy
import utils.wandb
import utils.botometer
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError, DuplicateKeyError, WriteError
import pymongo.collection
import tweepy
from tweepy.errors import TooManyRequests, TwitterServerError, HTTPException
from utils.converter import V2_TO_V1_TWEET, bend_user
from jsonbender import bend
from math import ceil
import wandb

from dotenv import load_dotenv

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

main_logger = logging.getLogger(__name__)

config = {
    "min_tweets": 200,
    "num_users": 1500,
    "num_timeline_tweets": 200,
    "num_mention_tweets": 100,
    "end_time": "2023-01-19T11:59:59Z",
}


def aggr_one(self, pipeline):
    cursor = self.aggregate(pipeline)
    return next(cursor, None)


pymongo.collection.Collection.aggregate_one = aggr_one


def match_in_collection(collection, condition):
    return len(list(collection.find(condition))) > 0


def request_error_handler(request_fn):
    """
    Handles exceptions commonly thrown by tweepy.Client requests. Can be used as a decorator on a function that calls the twitter API.


    Args:
        request_fn: The function that throws these exceptions.
    """

    def inner_fn(*args, **kwargs):
        try:
            return request_fn(*args, **kwargs)
        except TooManyRequests as e:
            res = e.response
            main_logger.error(
                "(%s) HTTP-%d: %s. Too many requests."
                % (request_fn.__name__, res.status_code, res.reason)
            )
            main_logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))
            main_logger.info("Sleeping for 15 minutes.")
            time.sleep(60 * 15)
        except TwitterServerError as e:
            res = e.response
            main_logger.error(
                "(%s) HTTP-%d: %s. Twitter server error."
                % (request_fn.__name__, res.status_code, res.reason)
            )
            main_logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))
        except HTTPException as e:
            res = e.response
            main_logger.error(
                "(%s) HTTP-%d: %s" % (request_fn.__name__, res.status_code, res.reason)
            )
            main_logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))

    return inner_fn


@request_error_handler
def access_user_data(client, user_id):
    """
    Queries user data, and returns it.
    Errors can occur if the user profile has been set to private, has been suspended or deleted, etc.

    Args:
        api (tweepy.Client): Client for Twitter API.
        user_id (int | str): ID of user we want to query.

    Returns:
        dict: User object, None if user could not be retrieved.
    """
    response = client.get_user(id=user_id, user_fields=utils.tweepy.ALL_USER_FIELDS)
    user_obj = response.data
    if user_obj is not None:
        return user_obj.data
    return None


@request_error_handler
def get_user_tweets(client, user_id, limit, method="timeline", author=None, **kwargs):
    """
    Returns tweets from the specified user's timeline.

    Args:
        api (tweepy.Client): Client for Twitter API.
        user_id (int | str): ID of user we want to query.
        limit (int): Max number of tweets we want to pull.
        method (string): "timeline" or "mentions". Specifies what tweets to pull. Default is "timeline".
        author (dict | None): Author object of pulled tweets. Default is None.
        kwargs: Keyword arguments passed to tweepy.Paginator

    Returns:
        list: List of user's tweets (aka. their timeline).
        string: The next token for querying the timeline.

        A tweet is a dictionary with two keys - "data" and "includes" with the following structure:
                "data": Data related to tweet: id, created_at, author_id, text, etc...
                "includes":
                    "users": List of user objects related to tweet. First one is author of tweet.
    """
    timeline = []

    assert method in ["timeline", "mentions"]
    if method == "timeline":
        method_fn = client.get_users_tweets
    elif method == "mentions":
        method_fn = client.get_users_mentions

    if author is None:
        expansions = "author_id"
    else:
        expansions = None

    max_results = min(
        100, max(5, limit)
    )  # max results per page - 100 is the maximum, 5 is the minimum
    paginator = tweepy.Paginator(
        method=method_fn,
        id=user_id,
        expansions=expansions,
        tweet_fields=utils.tweepy.TWEET_PUBLIC_FIELDS,
        user_fields=utils.tweepy.ALL_USER_FIELDS,
        max_results=max_results,
        limit=ceil(limit / max_results),  # how many calls to make to the api
        **kwargs,
    )

    def author_from_includes(id, includes):
        return next(
            author.data for author in includes["users"] if author.data["id"] == id
        )

    for page in paginator:
        no_results = page.meta["result_count"] == 0
        next_token = None if not "next_token" in page.meta else page.meta["next_token"]
        if no_results:
            return timeline[:limit], next_token
        for tweet_obj in page.data:
            queried_at = datetime.utcnow().isoformat()
            tweet_obj.data["queried_at"] = queried_at
            if author is None:
                author = author_from_includes(
                    id=tweet_obj.data["author_id"], includes=page.includes
                )
            tweet = {
                "data": tweet_obj.data,
                "includes": {
                    "users": [author],
                },
            }
            timeline.append(tweet)
    return timeline[:limit], next_token


def count_retweets(timeline):
    num_RT = 0
    for tweet in timeline:
        has_referenced_tweets = "referenced_tweets" in tweet["data"]
        if has_referenced_tweets and "retweeted" in [
            ref["type"] for ref in tweet["data"]["referenced_tweets"]
        ]:
            num_RT += 1
    return num_RT


def duplicate_key_error_handler(insert_fn):
    """
    Handles DuplicateKeyError commonly thrown by pymongo collection inserts. Can be used as a decorator.

    Args:
        insert_fn: The function that throws these exceptions.
    """

    def inner_fn(*args, **kwargs):
        try:
            return insert_fn(*args, **kwargs)
        except DuplicateKeyError as e:
            main_logger.debug(e.details)
        except BulkWriteError as e:
            for write_error in e.details["writeErrors"]:
                # just log if duplicate key error occurs
                if write_error["code"] == 11000:
                    main_logger.debug(write_error)
                else:
                    raise WriteError(
                        error=write_error["errmsg"],
                        code=write_error["code"],
                    )

    return inner_fn


@duplicate_key_error_handler
def insert_many(collection, items):
    if len(items) > 0:
        insert_result = collection.insert_many(items)
        main_logger.debug(
            "Inserted %d items into %s collection."
            % (len(insert_result.inserted_ids), collection.name)
        )
    else:
        main_logger.warn("No items in list to insert into %s." % collection.name)


@duplicate_key_error_handler
def insert_one(collection, item):
    if item is not None:
        collection.insert_one(item)
    else:
        main_logger.warn("Item is None, can't insert into %s." % collection.name)


class SampledUser:
    def __init__(self, user):
        self._user_object = user
        self._metrics = [
            "num_pulled_tweets",
            "num_pulled_mentions",
            "num_RT_tweets",
            "majority_lang",
            "bot_score",
        ]
        self._init_sampled_user()

    def _init_sampled_user(self):
        self._sampled_user = {
            "id": self._user_object["id"],
            "username": self._user_object["username"],
            "created_at": self._user_object["created_at"],
            "tweet_count": self._user_object["public_metrics"]["tweet_count"],
            "error_or_no_access": None,
            "next_token": None,
        }
        for key in self._metrics:
            self._sampled_user[key] = None

    @property
    def id(self):
        return self._sampled_user["id"]

    @property
    def error_or_no_access(self):
        return self._sampled_user["error_or_no_access"]

    @error_or_no_access.setter
    def error_or_no_access(self, value: bool):
        self._sampled_user["error_or_no_access"] = value

    @property
    def tweet_num(self):
        return self._sampled_user["num_pulled_tweets"]

    @tweet_num.setter
    def tweet_num(self, value: int):
        self._sampled_user["num_pulled_tweets"] = value

    @property
    def mention_num(self):
        return self._sampled_user["num_pulled_mentions"]

    @mention_num.setter
    def mention_num(self, value: int):
        self._sampled_user["num_pulled_mentions"] = value

    @property
    def RT_num(self):
        return self._sampled_user["num_RT_tweets"]

    @RT_num.setter
    def RT_num(self, value: int):
        num_tweets = self._sampled_user["num_pulled_tweets"]
        if num_tweets is not None:
            assert value <= num_tweets

        self._sampled_user["num_RT_tweets"] = value

    @property
    def majority_lang(self):
        return self._sampled_user["majority_lang"]

    @majority_lang.setter
    def majority_lang(self, value: str):
        self._sampled_user["majority_lang"] = value

    @property
    def bot_score(self):
        return self._sampled_user["bot_score"]

    @bot_score.setter
    def bot_score(self, value: dict):
        self._sampled_user["bot_score"] = value

    def _is_populated(self) -> bool:
        for key in self._metrics:
            if self._sampled_user[key] is None:
                return False
        return True

    def save_to_collection(self, collection):
        if self.error_or_no_access is None:
            raise RuntimeError(
                "Could not save subject (id: %s). error_or_no_access not set."
                % (self.id)
            )
        else:
            if not self._is_populated():
                main_logger.warning(
                    "User with id %s was not populated before saving." % self.id
                )
            insert_one(collection, self._sampled_user)

    @property
    def next_token(self):
        return self._sampled_user["next_token"]

    @next_token.setter
    def next_token(self, value: str):
        self._sampled_user["next_token"] = value


if __name__ == "__main__":
    # wandb
    utils.wandb.init_wandb_run(job_type="sample-users", config=config)
    cfg = wandb.config

    # logging
    utils.logging.log_to_stdout("main", level=logging.DEBUG)
    utils.logging.log_to_stdout("utils", level=logging.DEBUG)
    utils.logging.log_to_stdout("tweepy", level=logging.DEBUG)

    # setup MongoDB
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database
    tweets_collection = db.tweets_collection
    users_collection = db.users_collection

    # collections where we will save our results
    sampled_users_collection = db.sampled_users_collection
    sampled_users_collection.create_index([("id", ASCENDING)], unique=True)

    timelines_collection = db.timelines_collection
    timelines_collection.create_index([("id", ASCENDING)], unique=True)
    timelines_collection.create_index([("author_id", ASCENDING)])

    mentions_collection = db.mentions_collection
    mentions_collection.create_index(
        [("mentioned_user_id", ASCENDING), ("id", ASCENDING)], unique=True
    )  # compound index on id and mentioned_user_id

    # setup tweepy client
    client = tweepy.Client(bearer_token=os.environ["BEARER_TOKEN"])

    one_random_sample = {
        "$sample": {"size": 1},
    }

    num_users = 0
    while num_users < cfg["num_users"]:
        rand_tweet = tweets_collection.aggregate_one(pipeline=[one_random_sample])

        user = users_collection.find_one({"id": rand_tweet["author_id"]})
        tweet_count = user["public_metrics"]["tweet_count"]

        if tweet_count < cfg["min_tweets"] or match_in_collection(
            sampled_users_collection, {"id": user["id"]}
        ):
            pass
        else:
            num_users += 1
            main_logger.info("Processing %d. user..." % num_users)
            sampled_user = SampledUser(user)
            # check if we can access user data
            # --> error_on_access
            v2_user = access_user_data(client=client, user_id=sampled_user.id)
            sampled_user.error_or_no_access = v2_user is None or v2_user["protected"]
            if sampled_user.error_or_no_access:
                main_logger.info(
                    "User data cannot be accessed. Id: %s" % sampled_user.id
                )
                sampled_user.save_to_collection(sampled_users_collection)
            else:
                # collect 200 tweets
                # --> RT_ratio
                v2_user_timeline, next_token = get_user_tweets(
                    client=client,
                    user_id=sampled_user.id,
                    limit=cfg["num_timeline_tweets"],
                    method="timeline",
                    author=v2_user,
                    end_time=cfg["end_time"],
                )
                sampled_user.tweet_num = len(v2_user_timeline)
                sampled_user.next_token = next_token
                if sampled_user.tweet_num == 0:
                    main_logger.info(
                        "No tweets found on user timeline before end date. Id: %s"
                        % sampled_user.id
                    )
                    sampled_user.save_to_collection(sampled_users_collection)
                    continue

                sampled_user.RT_num = count_retweets(timeline=v2_user_timeline)

                # collect 100 mentions
                v2_user_mentions, _ = get_user_tweets(
                    client=client,
                    user_id=sampled_user.id,
                    limit=cfg["num_mention_tweets"],
                    method="mentions",
                    end_time=cfg["end_time"],
                )
                sampled_user.mention_num = len(v2_user_mentions)

                # convert tweets and user to v1 object model
                def bend_tweet(v2_tweet):
                    return bend(mapping=V2_TO_V1_TWEET, source=v2_tweet)

                v1_user_timeline = list(map(bend_tweet, v2_user_timeline))
                v1_user_mentions = list(map(bend_tweet, v2_user_mentions))
                v1_user = bend_user(v2_user)

                # Bot-O-Meter request
                # --> majority_lang
                # --> bot_score
                response = utils.botometer.botometer_request(
                    timeline=v1_user_timeline, mentions=v1_user_mentions, user=v1_user
                )
                body = response.json()
                sampled_user.majority_lang = body["user"]["majority_lang"]
                sampled_user.bot_score = {
                    "cap": body["cap"],
                    "raw_scores": body["raw_scores"],
                }

                # save random user
                sampled_user.save_to_collection(sampled_users_collection)

                # save timeline and mentions
                insert_many(
                    timelines_collection, [tweet["data"] for tweet in v2_user_timeline]
                )
                insert_many(
                    mentions_collection,
                    [
                        dict(tweet["data"], **{"mentioned_user_id": sampled_user.id})
                        for tweet in v2_user_mentions
                    ],
                )

    # close connections
    mongo_conn.close()
