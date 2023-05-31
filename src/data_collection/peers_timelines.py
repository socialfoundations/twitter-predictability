from dotenv import load_dotenv
import logging, os
from pymongo import MongoClient, ASCENDING
import utils.wandb
import utils.logging
import utils.tweepy
import utils.mongo
import wandb
import tweepy
from collections import Counter
from datetime import datetime

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

# main logger
main_logger = logging.getLogger("main")

# config
config = {
    "database": "twitter",
    "subject_tweets": 500,
    "peers_per_subject": 15,
    "tweets_per_peer": 50,
    "exclude": ["retweets"],
    "end_time": "2023-01-19T11:59:59Z",  # don't collect tweets after this timestamp
    "filter": {"lang": "en"},
    "method": "timeline",  # method with which we pull the tweets
}

if __name__ == "__main__":
    # wandb - "offline" mode to avoid sending wandb runs which are constantly failing
    utils.wandb.init_wandb_run(
        job_type="peer-timelines",
        config=config,
        mode="online",
        log_code=True,
        # tags=["debug"],
    )
    cfg = wandb.config

    # set loggers to lower or same level as handlers
    utils.logging.set_logger_levels(
        ["main", "utils", "tweepy", "retry"], level=logging.DEBUG
    )

    # logging handlers - INFO to stdout and DEBUG to file
    utils.logging.logs_to_stdout(
        ["main", "utils", "tweepy", "retry"], level=logging.INFO
    )
    utils.logging.logs_to_file(
        ["main", "utils", "tweepy", "retry"], logdir=wandb.run.dir, level=logging.DEBUG
    )

    # setup MongoDB
    main_logger.info("Connecting to %s database..." % cfg["database"])
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn[cfg["database"]]  # our database

    users_collection = db["users_collection"]
    subjects_collection = db["subjects_collection"]
    timelines_collection = db["timelines_collection"]

    # create peers collection
    peers_collection = db["peers_collection"]
    peers_collection.create_index([("id", ASCENDING)], unique=True)

    # setup tweepy client
    main_logger.info("Connecting to Twitter API...")
    client = tweepy.Client(bearer_token=os.environ["BEARER_TOKEN"])

    with mongo_conn.start_session() as session:
        cursor = subjects_collection.find(
            {"timeline_tweets_count": {"$gte": cfg["subject_tweets"]}},
            no_cursor_timeout=True,  # make sure there is a .close() at the end!
            session=session,
        )

        for i, s in enumerate(cursor):
            subject_id = s["id"]
            main_logger.info("Processing %d, user with id %s..." % (i, subject_id))

            user_timeline_no_RT_en = {
                "author_id": subject_id,
                "referenced_tweets.0.type": {"$ne": "retweeted"},
                "lang": "en",
            }

            mentions = Counter()

            t_cursor = timelines_collection.find(
                user_timeline_no_RT_en,
                session=session,
            ).limit(cfg["subject_tweets"])

            for t in t_cursor:
                if "entities" in t and "mentions" in t["entities"]:
                    mentioned_ids = [
                        mention["id"] for mention in t["entities"]["mentions"]
                    ]
                    mentions = mentions + Counter(mentioned_ids)
            t_cursor.close()

            # top-K mentioned users
            top_mentioned = mentions.most_common(cfg["peers_per_subject"])

            # for each frequently mentioned peer, collect timeline
            for rank, (mentioned_id, n_mentions) in enumerate(top_mentioned, start=1):
                main_logger.info(
                    "-- Processing rank-%d peer with %d mentions: %s"
                    % (rank, n_mentions, mentioned_id)
                )
                # skip collection for users that are already in the peers collection and have enough tweets
                match = peers_collection.find_one(
                    {"id": mentioned_id},
                    session=session,
                )
                if match:
                    main_logger.debug(
                        "-- Peer already processed. Updating mentioned_by list with %s."
                        % subject_id
                    )
                    # add subject to mentioned_by list
                    peers_collection.update_one(
                        {"id": mentioned_id},
                        {
                            "$addToSet": {
                                "mentioned_by": {
                                    "id": subject_id,
                                    "rank": rank,
                                    "num_mentions": n_mentions,
                                }
                            }
                        },
                    )
                    # continue
                    # skip collection if enough tweets
                    if match["timeline_tweets_count"] >= cfg["tweets_per_peer"]:
                        continue

                # skip users that have been deleted / set to private / are protected etc.
                result = utils.tweepy.access_user_data(client, user_id=mentioned_id)
                if result is None or result["protected"]:
                    main_logger.warning(
                        "-- Peer data could not be accessed. Skipping user."
                    )
                    continue
                else:
                    # add to user collection if not in it
                    if not utils.mongo.match_in_collection(
                        users_collection, {"id": mentioned_id}
                    ):
                        main_logger.debug(
                            "-- Peer not yet in users collection. Inserting..."
                        )
                        result["queried_at"] = datetime.utcnow().isoformat()
                        utils.mongo.insert_one(users_collection, result)

                # look at how many tweets we have from the peer already
                user_timeline_no_RT_en["author_id"] = mentioned_id
                n_tweets = timelines_collection.count_documents(
                    user_timeline_no_RT_en,
                    session=session,
                )
                main_logger.debug(
                    "-- %d tweets in collection from %s." % (n_tweets, mentioned_id)
                )

                if n_tweets < cfg["tweets_per_peer"]:
                    num = cfg["tweets_per_peer"] - n_tweets
                    main_logger.debug(
                        "-- Collecting %d tweets for %s..." % (num, mentioned_id)
                    )

                    if n_tweets == 0:
                        end_time = cfg["end_time"]
                    else:
                        oldest_tweet = timelines_collection.find_one(
                            user_timeline_no_RT_en,
                            sort=[("_id", -1)],
                            session=session,
                        )
                        end_time = oldest_tweet["created_at"]

                    # get the peer's timeline
                    timeline, _ = utils.tweepy.get_user_tweets(
                        client,
                        method=cfg["method"],
                        user_id=mentioned_id,
                        minimum=num,
                        end_time=end_time,
                        exclude=cfg["exclude"],
                        filter_conditions=cfg["filter"],
                    )

                    # save it to timelines collection
                    utils.mongo.insert_many(
                        timelines_collection, [tweet["data"] for tweet in timeline]
                    )
                else:
                    main_logger.debug(
                        "-- No further tweet collection needed for %s." % mentioned_id
                    )

                # number of non-RT tweets after collection
                n_tweets = timelines_collection.count_documents(
                    user_timeline_no_RT_en,
                    session=session,
                )
                main_logger.debug(
                    "-- %d tweets in collection from %s." % (n_tweets, mentioned_id)
                )

                # save user into peer collection
                user = users_collection.find_one(
                    {"id": mentioned_id},
                    session=session,
                )
                user["timeline_tweets_count"] = n_tweets
                user["mentioned_by"] = [
                    {"id": subject_id, "rank": rank, "num_mentions": n_mentions}
                ]

                if utils.mongo.match_in_collection(
                    peers_collection, {"id": mentioned_id}
                ):
                    peers_collection.update_one(
                        {"id": mentioned_id},
                        {
                            "$addToSet": {
                                "mentioned_by": {
                                    "id": subject_id,
                                    "rank": rank,
                                    "num_mentions": n_mentions,
                                }
                            },
                            "$set": {"timeline_tweets_count": n_tweets},
                        },
                    )
                else:
                    utils.mongo.insert_one(peers_collection, user)

        # cleanup
        cursor.close()
        mongo_conn.close()
