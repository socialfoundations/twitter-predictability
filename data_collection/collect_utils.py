import logging, os
from ssl_smtp_handler import SSLSMTPHandler

ALL_EXPANSIONS = [
    "geo.place_id",
    "author_id",
    "attachments.media_keys",
    "attachments.poll_ids",
]

ALL_TWEET_FIELDS = [
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
    "non_public_metrics",
    "organic_metrics",
    "possibly_sensitive",
    "promoted_metrics",
    "public_metrics",
    "referenced_tweets",
    "reply_settings",
    "source",
    "withheld",
]

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
