import tweepy
import tweepy.errors
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError, WriteError
import os, logging, time
from datetime import datetime
from tqdm import tqdm
import uuid
import collect_utils
from collect_utils import main_logger, log_to_file, get_email_logger

from dotenv import load_dotenv

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

# maximum number of tweets to pull
MAX_TWEETS = 1e7


class TweetSaverClient(tweepy.StreamingClient):
    """A class that saves tweets into our MongoDB collection."""

    def __init__(
        self,
        tweets_collection,
        users_collection,
        places_collection,
        media_collection,
        polls_collection,
        *args,
        **kwargs
    ):
        super(TweetSaverClient, self).__init__(*args, **kwargs)
        self.tweets_collection = tweets_collection
        self.users_collection = users_collection
        self.places_collection = places_collection
        self.media_collection = media_collection
        self.polls_collection = polls_collection

    def on_tweet(self, tweet):
        data = tweet.data
        data["queried_at"] = datetime.utcnow().isoformat()
        self.tweets_collection.insert_one(data)

    def insert(self, collection, data):
        try:
            collection.insert_many(data, ordered=False)
        except BulkWriteError as e:
            for write_error in e.details["writeErrors"]:
                # just log if duplicate key error occurs
                if write_error["code"] == 11000:
                    main_logger.debug(write_error)
                else:
                    raise WriteError(
                        error=write_error["errmsg"], code=write_error["code"]
                    )

    def on_includes(self, includes):
        queried_at = datetime.utcnow().isoformat()
        keys = includes.keys()
        if "users" in keys:
            user_data = [user.data for user in includes["users"]]
            self.insert(self.users_collection, user_data)
        if "places" in keys:
            place_data = [place.data for place in includes["places"]]
            self.insert(self.places_collection, place_data)
        if "media" in keys:
            media_data = [
                dict(media.data, **{"queried_at": queried_at})
                for media in includes["media"]
            ]
            self.insert(self.media_collection, media_data)
        if "polls" in keys:
            polls_data = [
                dict(poll.data, **{"queried_at": queried_at})
                for poll in includes["polls"]
            ]
            self.insert(self.polls_collection, polls_data)

    def on_closed(self, response):
        main_logger.error("Closed stream with response: %s" % response)
        return super().on_closed(response)

    def on_exception(self, exception):
        main_logger.error("Stream encountered exception: %s" % exception)
        return super().on_exception(exception)


def stream_loop(streaming_client, count, max_count, restart, tweets_collection):
    if not restart:
        main_logger.info("Starting stream.")
        backfill_minutes = None
    else:
        main_logger.info("Re-starting stream.")
        backfill_minutes = (
            5  # to make up for the data that we might have missed during the downtime
        )

    streaming_client.filter(
        threaded=True,
        expansions=collect_utils.ALL_EXPANSIONS,
        tweet_fields=collect_utils.ALL_TWEET_FIELDS,
        user_fields=collect_utils.ALL_USER_FIELDS,
        place_fields=collect_utils.ALL_PLACE_FIELDS,
        media_fields=collect_utils.ALL_MEDIA_FIELDS,
        poll_fields=collect_utils.ALL_POLL_FIELDS,
        backfill_minutes=backfill_minutes,
    )

    with tqdm(total=max_count) as pbar:
        pbar.update(count)
        while streaming_client.running and count < max_count:
            # check number of tweets every so often
            time.sleep(60 * 5)
            new_count = tweets_collection.estimated_document_count()
            main_logger.info("Number of tweets in database: %d " % new_count)
            delta = new_count - count
            pbar.update(delta)
            count = new_count

    return count


if __name__ == "__main__":
    # setup logging
    run_id = str(uuid.uuid1())
    log_path = os.path.join("logs", run_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_to_file(
        "main", os.path.join(log_path, "collect_tweets.log"), level=logging.INFO
    )
    log_to_file("tweepy", os.path.join(log_path, "tweepy.log"), level=logging.INFO)
    email_logger = get_email_logger(subject="collect_tweets.py")
    email_logger.info("Started running. UUID: %s" % run_id)

    # setup MongoDB
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database
    tweets_collection = db.tweets_collection
    tweets_collection.create_index(
        [("id", ASCENDING)], unique=True
    )  # create index on tweet id
    users_collection = db.users_collection
    users_collection.create_index([("id", ASCENDING)], unique=True)
    places_collection = db.places_collection
    places_collection.create_index([("id", ASCENDING)], unique=True)
    media_collection = db.media_collection
    media_collection.create_index([("media_key", ASCENDING)], unique=True)
    polls_collection = db.polls_collection
    polls_collection.create_index([("id", ASCENDING)], unique=True)

    # setup Twitter API client
    streaming_client = TweetSaverClient(
        tweets_collection=tweets_collection,
        users_collection=users_collection,
        places_collection=places_collection,
        media_collection=media_collection,
        polls_collection=polls_collection,
        bearer_token=os.environ["BEARER_TOKEN"],
        wait_on_rate_limit=True,
    )

    # clear all rules that were there before
    response = streaming_client.get_rules()
    if response.meta["result_count"] != 0:
        rule_ids = [rule.id for rule in response.data]
        streaming_client.delete_rules(ids=rule_ids)

    # add filtering rule
    stream_rule_string = "sample:1 followers_count:0 -is:retweet lang:en"
    response = streaming_client.add_rules(tweepy.StreamRule(stream_rule_string))
    main_logger.info(response)

    # stream data
    restart = False
    count = tweets_collection.estimated_document_count()
    while count < MAX_TWEETS:
        count = stream_loop(
            streaming_client,
            count,
            max_count=MAX_TWEETS,
            restart=restart,
            tweets_collection=tweets_collection,
        )
        if count < MAX_TWEETS:
            email_logger.warning(
                "Streaming client stopped running before we reached the desired number of tweets: %d / %d"
                % (count, MAX_TWEETS)
            )
            restart = True

    # after finishing, disconnect
    if streaming_client.running:
        streaming_client.disconnect()

    main_logger.info(
        "Finished running. Current number of tweets in database: %d" % (count)
    )
    email_logger.info("Finished running. UUID: %s" % run_id)
