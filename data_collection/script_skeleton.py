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

    # logging - DEBUG is encouraged while writing the script, later this can be changed to INFO and writing logs to file
    utils.logging.log_to_stdout("main", level=logging.DEBUG)
    utils.logging.log_to_stdout("utils", level=logging.DEBUG)
    utils.logging.log_to_stdout("tweepy", level=logging.DEBUG)

    # setup MongoDB
    main_logger.info("Connecting to %s database..." % cfg["database"])
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn[cfg["database"]]  # our database

    # setup tweepy client
    main_logger.info("Connecting to Twitter API...")
    client = tweepy.Client(bearer_token=os.environ["BEARER_TOKEN"])
