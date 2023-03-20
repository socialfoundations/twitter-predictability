import os, logging
import utils.logging
import utils.tweepy
import utils.wandb
import utils.botometer
import utils.mongo
from pymongo import MongoClient, ASCENDING
import tweepy
from utils.converter import V2_TO_V1_TWEET, bend_user
from jsonbender import bend
import wandb

from dotenv import load_dotenv

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

main_logger = logging.getLogger("main")

config = {
    "min_tweets": 200,
    "num_users": 1500,
    "num_timeline_tweets": 200,
    "num_mention_tweets": 100,
    "end_time": "2023-01-19T11:59:59Z",
}


def count_retweets(timeline):
    num_RT = 0
    for tweet in timeline:
        has_referenced_tweets = "referenced_tweets" in tweet["data"]
        if has_referenced_tweets and "retweeted" in [
            ref["type"] for ref in tweet["data"]["referenced_tweets"]
        ]:
            num_RT += 1
    return num_RT


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
            utils.mongo.insert_one(collection, self._sampled_user)

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
    utils.logging.log_to_stdout("main", level=logging.INFO)
    utils.logging.log_to_stdout("utils", level=logging.INFO)
    utils.logging.log_to_stdout("tweepy", level=logging.INFO)

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

        if tweet_count < cfg["min_tweets"] or utils.mongo.match_in_collection(
            sampled_users_collection, {"id": user["id"]}
        ):
            pass
        else:
            num_users += 1
            main_logger.info("Processing %d. user..." % num_users)
            sampled_user = SampledUser(user)
            # check if we can access user data
            # --> error_on_access
            v2_user = utils.tweepy.access_user_data(
                client=client, user_id=sampled_user.id
            )
            sampled_user.error_or_no_access = v2_user is None or v2_user["protected"]
            if sampled_user.error_or_no_access:
                main_logger.info(
                    "User data cannot be accessed. Id: %s" % sampled_user.id
                )
                sampled_user.save_to_collection(sampled_users_collection)
            else:
                # collect 200 tweets
                # --> RT_ratio
                v2_user_timeline, next_token = utils.tweepy.get_user_tweets(
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
                v2_user_mentions, _ = utils.tweepy.get_user_tweets(
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
                utils.mongo.insert_many(
                    timelines_collection, [tweet["data"] for tweet in v2_user_timeline]
                )
                utils.mongo.insert_many(
                    mentions_collection,
                    [
                        dict(tweet["data"], **{"mentioned_user_id": sampled_user.id})
                        for tweet in v2_user_mentions
                    ],
                )

    # close connections
    mongo_conn.close()
