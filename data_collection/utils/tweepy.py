import logging, time
from datetime import datetime
import tweepy
from tweepy.errors import TooManyRequests, TwitterServerError, HTTPException
from math import ceil

logger = logging.getLogger(__name__)

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
            logger.error(
                "(%s) HTTP-%d: %s. Too many requests."
                % (request_fn.__name__, res.status_code, res.reason)
            )
            logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))
            logger.info("Sleeping for 15 minutes.")
            time.sleep(60 * 15)
        except TwitterServerError as e:
            res = e.response
            logger.error(
                "(%s) HTTP-%d: %s. Twitter server error."
                % (request_fn.__name__, res.status_code, res.reason)
            )
            logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))
        except HTTPException as e:
            res = e.response
            logger.error(
                "(%s) HTTP-%d: %s" % (request_fn.__name__, res.status_code, res.reason)
            )
            logger.debug("Header: %s\nContent: %s" % (res.headers, res.content))

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
    response = client.get_user(id=user_id, user_fields=ALL_USER_FIELDS)
    user_obj = response.data
    if user_obj is not None:
        return user_obj.data
    return None


def _tweet_fits(conditions: dict, tweet):
    """
    A function which tells us whether the given tweet fits all of the specified conditions.
    It assumes that the parameter exists. It doesn't work on nested parameters like "public_metrics.retweet_count".

    Args:
        conditions (dict): A dict of param - value pairs. It specifies what value the params have to take. None if no filtering is needed.
        tweet (dict): The tweet we are testing.

    Returns:
        bool: Whether or not the tweet fits the conditions.
    """
    if conditions is None:
        return True

    for key, val in conditions.items():
        if tweet[key] != val:
            return False
    return True


class PaginatorConfig:
    def __init__(self, user_id, minimum, method_fn, expansions, **kwargs):
        self.user_id = user_id

        self.max_results = min(
            100, max(5, minimum)
        )  # max results per page - 100 is the maximum, 5 is the minimum

        self.method_fn = method_fn
        self.expansions = expansions
        self.kwargs = kwargs


def create_paginator(config: PaginatorConfig, num_tweets, next_token=None):
    """
    Returns a paginator that will try to fetch num_tweets tweets.

    Args:
        config (PaginatorConfig): Parameters for instantiating the Paginator.
        num_tweets (int): How many tweets we want to pull.
        next_token (str | None): If set, collect tweets starting from this token. Defaults to None.

    Returns:
        tweepy.Paginator: The created paginator.
    """
    paginator = tweepy.Paginator(
        method=config.method_fn,
        id=config.user_id,
        expansions=config.expansions,
        tweet_fields=TWEET_PUBLIC_FIELDS,
        user_fields=ALL_USER_FIELDS,
        max_results=config.max_results,  # results per page
        limit=ceil(num_tweets / config.max_results) + 1,  # number of pages
        pagination_token=next_token,
        **config.kwargs,
    )

    return paginator


@request_error_handler
def collect_and_filter(paginator, filter_conditions, author=None):
    """
    Iterates over paginator, and collects tweets that fit the filter condition.
    If author is not set and page.includes["users"] includes tweet author, it uses that instead.

    Args:
        paginator (tweepy.Paginator): The paginator.
        filter_conditions (dict): Dictionary of conditions that the pulled tweets have to satisfy. None if no filtering is needed.
        author (dict | None): Author object of pulled tweets. Defaults to None.

    Returns:
        list: List of user's tweets (aka. their timeline).
        string: The next token for querying the timeline.

        A tweet is a dictionary with two keys - "data" and "includes" with the following structure:
                "data": Data related to tweet: id, created_at, author_id, text, etc...
                "includes":
                    "users": List of user objects related to tweet. First one is author of tweet.
    """
    timeline = []

    def author_from_includes(id, includes):
        if "users" in includes:
            return next(
                author.data for author in includes["users"] if author.data["id"] == id
            )
        else:
            return None

    for page in paginator:
        no_results = page.meta["result_count"] == 0
        next_token = None if not "next_token" in page.meta else page.meta["next_token"]
        if no_results:
            return timeline, next_token
        for tweet_obj in filter(
            lambda tweet_obj: _tweet_fits(filter_conditions, tweet_obj.data), page.data
        ):
            queried_at = datetime.utcnow().isoformat()
            tweet_obj.data["queried_at"] = queried_at
            if author is None and page.includes:
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

    return timeline, next_token


def get_user_tweets(
    client,
    user_id,
    minimum,
    method="timeline",
    author=None,
    filter_conditions=None,
    max_retries=3,
    **kwargs
):
    """
    Returns tweets from the specified user's timeline.

    Args:
        client (tweepy.Client): Client for Twitter API.
        user_id (int | str): ID of user we want to query.
        minimum (int): Number of tweets we want to pull (returned number of tweets might be higher).
        method (string): "timeline" or "mentions". Specifies what tweets to pull. Default is "timeline".
        filter_conditions (dict): A dict of param - value pairs. Specifies conditions that the tweet has to fit.
        author (dict | None): Author object of pulled tweets. Default is None.
        filter_conditions (dict | None): A dict of param - value pairs. It specifies what value the params have to take. None if no filtering is needed. Default is None.
        kwargs: Keyword arguments passed to tweepy.Paginator

    Returns:
        list: List of user's tweets (aka. their timeline).
        string: The next token for querying the timeline.

        A tweet is a dictionary with two keys - "data" and "includes" with the following structure:
                "data": Data related to tweet: id, created_at, author_id, text, etc...
                "includes":
                    "users": List of user objects related to tweet. First one is author of tweet.
    """
    method_fn = (
        client.get_users_mentions if method == "mentions" else client.get_users_tweets
    )
    expansions = None if author is not None else "author_id"

    paginator_config = PaginatorConfig(
        user_id=user_id,
        minimum=minimum,
        method_fn=method_fn,
        expansions=expansions,
        **kwargs,
    )

    paginator = create_paginator(
        config=paginator_config,
        num_tweets=minimum,
    )

    timeline, next_token = collect_and_filter(
        paginator, filter_conditions, author=author
    )

    # if too many tweets were filtered out
    for i in range(max_retries):
        if len(timeline) < minimum and next_token is not None:
            missing = minimum - len(timeline)
            logger.debug(
                "Retry %d... Collected: %d Missing: %d" % (i, len(timeline), missing)
            )
            # create new paginator
            paginator = create_paginator(
                config=paginator_config, num_tweets=missing, next_token=next_token
            )
            tweets, next_token = collect_and_filter(
                paginator, filter_conditions, author=author
            )
            timeline.extend(tweets)

    if len(timeline) < minimum:
        logger.warning(
            "Collected less tweets than expected: %d (minimum: %d)."
            % (len(timeline), minimum)
        )

    return timeline, next_token
