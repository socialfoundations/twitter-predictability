from dotenv import load_dotenv
import logging, os
from pymongo import MongoClient, ASCENDING
import utils.wandb
import utils.logging
import utils.tweepy
import utils.mongo
import wandb
import tweepy

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

# main logger
main_logger = logging.getLogger("main")

# config
config = {
    "database": "twitter",
    "num_subjects": 100,
    "max_rt_ratio": 0.8,
    "max_bot_score": 0.5,
    "min_tweets": 500,  # in timeline, including retweets!
    "tweets_per_subject": 500,  # how many tweets to try collect that match specifications
    "exclude": ["retweets"],
    "end_time": "2023-01-19T11:59:59Z",  # don't collect tweets after this timestamp
    "filter": {"lang": "en"},
}


if __name__ == "__main__":
    # wandb
    utils.wandb.init_wandb_run(
        job_type="subject-timelines",
        config=config,
        mode="online",
        log_code=True,
        # tags=["debug"],
    )
    cfg = wandb.config

    # logging
    utils.logging.log_to_stdout("main", level=logging.INFO)
    utils.logging.log_to_stdout("utils", level=logging.INFO)
    utils.logging.log_to_stdout("tweepy", level=logging.INFO)

    # setup MongoDB
    main_logger.info("Connecting to %s database..." % cfg["database"])
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn[cfg["database"]]  # our database

    users_collection = db["users_collection"]
    sampled_users_collection = db["sampled_users_collection"]
    timelines_collection = db["timelines_collection"]

    # create subjects collection
    subjects_collection = db["subjects_collection"]
    subjects_collection.create_index([("id", ASCENDING)], unique=True)

    # setup tweepy client
    main_logger.info("Connecting to Twitter API...")
    client = tweepy.Client(bearer_token=os.environ["BEARER_TOKEN"])

    set_RT_ratio = {
        "$set": {"RT_ratio": {"$divide": ["$num_RT_tweets", "$num_pulled_tweets"]}},
    }

    subjects_filter = {
        "$match": {
            "error_or_no_access": False,
            "majority_lang": "en",
            "num_pulled_tweets": {"$gt": 0},
            "tweet_count": {"$gt": cfg["min_tweets"]},
            "RT_ratio": {"$lt": cfg["max_rt_ratio"]},
            "bot_score.raw_scores.english.overall": {"$lt": cfg["max_bot_score"]},
        }
    }

    limit = {"$limit": cfg["num_subjects"]}

    cursor = sampled_users_collection.aggregate([set_RT_ratio, subjects_filter, limit])

    for i, subject in enumerate(cursor):
        main_logger.info("Processing %d, user with id %s..." % (i, subject["id"]))

        # skip users that have been deleted / set to private, etc.
        if utils.tweepy.access_user_data(client, user_id=subject["id"]) is None:
            main_logger.warning("User data could not be accessed. Skipping user.")
            continue

        # Check if we have the user's timeline in our collection already.
        # If yes how many tweets that satisfy our conditions (not RT, english)
        user_timeline_no_RT_en = {
            "author_id": subject["id"],
            "referenced_tweets.0.type": {"$ne": "retweeted"},
            "lang": "en",
        }
        n_tweets = timelines_collection.count_documents(user_timeline_no_RT_en)
        main_logger.debug(
            "%d tweets in collection from %s." % (n_tweets, subject["id"])
        )

        if n_tweets < cfg["tweets_per_subject"]:
            num = cfg["tweets_per_subject"] - n_tweets
            main_logger.debug("Collecting %d tweets for %s..." % (num, subject["id"]))

            if n_tweets == 0:
                until_id = None
            else:
                oldest_tweet = timelines_collection.find_one(
                    user_timeline_no_RT_en, sort=[("_id", -1)]
                )
                until_id = oldest_tweet["id"]

            # pull user timeline
            timeline, _ = utils.tweepy.get_user_tweets(
                client,
                user_id=subject["id"],
                minimum=num,
                end_time=cfg["end_time"],
                exclude=cfg["exclude"],
                until_id=until_id,
                filter_conditions=cfg["filter"],
            )

            # save it to timelines collection
            utils.mongo.insert_many(
                timelines_collection, [tweet["data"] for tweet in timeline]
            )
        else:
            main_logger.debug(
                "No further tweet collection needed for %s." % subject["id"]
            )

        # number of non-RT tweets after collection
        n_tweets = timelines_collection.count_documents(user_timeline_no_RT_en)
        main_logger.info("%d tweets in collection from %s." % (n_tweets, subject["id"]))

        # save user into subjects collection if enough tweets were collected
        if n_tweets >= cfg["tweets_per_subject"]:
            user = users_collection.find_one({"id": subject["id"]})
            utils.mongo.insert_one(subjects_collection, user)

    # cleanup
    mongo_conn.close()
