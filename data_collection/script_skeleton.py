##
# This is a skeleton that serves as a starting point for our data collection scripts.
##

from dotenv import load_dotenv
import logging, os
from pymongo import MongoClient
import utils.wandb
import utils.logging
import wandb
import tweepy

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

# main logger
main_logger = logging.getLogger("main")

# config
config = {
    "database": "twitter_demo",
}


if __name__ == "__main__":
    # wandb - "offline" mode to avoid sending wandb runs which are constantly failing
    utils.wandb.init_wandb_run(
        job_type="test-job", config=config, mode="offline", log_code=False
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

    # setup tweepy client
    main_logger.info("Connecting to Twitter API...")
    client = tweepy.Client(bearer_token=os.environ["BEARER_TOKEN"])

    # cleanup
    mongo_conn.close()
