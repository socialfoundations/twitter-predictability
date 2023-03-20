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
        tweet_fields=TWEET_PUBLIC_FIELDS,
        user_fields=ALL_USER_FIELDS,
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
