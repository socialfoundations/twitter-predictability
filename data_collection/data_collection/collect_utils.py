import warnings

warnings.warn(
    "Please use the utils module from now on!",
    category=DeprecationWarning,
    stacklevel=2,
)

import logging, os, sys
from ssl_smtp_handler import SSLSMTPHandler
import requests
import wandb

# main logger
main_logger = logging.getLogger("main")

ALL_EXPANSIONS = [
    "geo.place_id",
    "author_id",
    "attachments.media_keys",
    "attachments.poll_ids",
]

TWEET_PUBLIC_FIELDS = [
    "attachments",
    "author_id",
    "context_annotations",
    "conversation_id",
    "created_at",
    "edit_controls",
    "edit_history_tweet_ids",
    "entities",
    "geo",
    "in_reply_to_user_id",
    "lang",
    "possibly_sensitive",
    "public_metrics",
    "referenced_tweets",
    "reply_settings",
    "source",
    "withheld",
]

# Non-public, organic, and promoted metrics are only available for Tweets that have been created within the last 30 days.
# https://developer.twitter.com/en/docs/twitter-api/metrics
TWEET_PRIVATE_FIELDS = [
    "non_public_metrics",
    "organic_metrics",
    "promoted_metrics",
]

ALL_TWEET_FIELDS = TWEET_PUBLIC_FIELDS + TWEET_PRIVATE_FIELDS


ALL_USER_FIELDS = [
    "created_at",
    "description",
    "entities",
    "location",
    "pinned_tweet_id",
    "profile_image_url",
    "protected",
    "public_metrics",
    "url",
    "verified",
    "verified_type",
    "withheld",
]

ALL_PLACE_FIELDS = [
    "contained_within",
    "country",
    "country_code",
    "geo",
    "name",
    "place_type",
]

ALL_MEDIA_FIELDS = [
    "alt_text",
    "duration_ms",
    "height",
    "non_public_metrics",
    "organic_metrics",
    "preview_image_url",
    "promoted_metrics",
    "public_metrics",
    "url",
    "variants",
    "width",
]

ALL_POLL_FIELDS = ["duration_minutes", "end_datetime", "voting_status"]


def log_to_stdout(logger_name, level=None):
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_to_file(logger_name, logfile, level=None):
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    handler = logging.FileHandler(filename=logfile, mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_email_logger(subject):
    email_logger = logging.getLogger("email")
    email_logger.setLevel("INFO")

    handler = SSLSMTPHandler(
        mailhost="smtp.gmail.com",
        fromaddr=os.environ["EMAIL_FROM"],
        toaddrs=os.environ["EMAIL_TO"],
        credentials=(os.environ["EMAIL_FROM"], os.environ["EMAIL_PASSWORD"]),
        subject=subject,
    )
    handler.setLevel("INFO")
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    email_logger.addHandler(handler)
    return email_logger


BOT_O_METER_URL = "https://botometer-pro.p.rapidapi.com/4/check_account"


def botometer_request(timeline, mentions, user):
    api_key = os.environ["RAPID_API_KEY"]
    if api_key is None:
        raise RuntimeError("No RapidAPI key provided.")

    if len(timeline) == 0:
        raise RuntimeError("No tweets in timeline.")

    payload = {
        "mentions": mentions,
        "timeline": timeline,
        "user": user,
    }

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": api_key,
    }

    main_logger.debug(
        "Making Bot-O-Meter request for %s user: based on %d tweets in timeline and %d mentions."
        % (user["id_str"], len(timeline), len(mentions))
    )
    response = requests.request("POST", BOT_O_METER_URL, json=payload, headers=headers)
    if response.status_code == 200:
        main_logger.debug(response.text)
        return response
    else:
        response.raise_for_status()


def init_wandb_run(config, job_type, mode="online"):
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity="social-foundations",
        save_code=True,
        job_type=job_type,
        config=config,
        mode=mode,
    )
    run.log_code()  # save code
