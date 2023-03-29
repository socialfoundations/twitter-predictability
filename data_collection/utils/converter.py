# Converts V2 Twitter API object models to V1.1
from jsonbender import K, S, F, OptionalS, bend
from jsonbender.list_ops import ForallBend
from jsonbender.control_flow import If, Alternation
from datetime import datetime


default_profile_image_url = (
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
)


def v2_to_v1_date(datestr):
    d = datetime.fromisoformat(datestr.replace("Z", "+00:00"))
    return d.strftime("%a %b %d %X %z %Y")


# Filled fields with default value:
#   - favourites_count: 0
#   - lang: null (deprecated V1 field)
#   - default_profile: True
#       Context:
#           * Bot-O-Meter friends feature: fraction of users with default profile AND default picture
#           * No information on background / theme in V2 API
V2_TO_V1_USER = {
    "id": S("data", "id") >> F(int),
    "id_str": S("data", "id"),
    "name": S("data", "name"),
    "screen_name": S("data", "username"),
    "location": OptionalS("data", "location"),
    "description": S("data", "description"),
    "entities": {
        "url": {
            "urls": OptionalS("data", "entities", "url", "urls", default=[])
            >> ForallBend(
                {
                    "url": S("url"),
                    "expanded_url": S("expanded_url"),
                    "display_url": S("display_url"),
                    "indices": [S("start"), S("end")],
                }
            ),
        },
    },
    "url": OptionalS("data", "url"),
    "protected": S("data", "protected"),
    "followers_count": S("data", "public_metrics", "followers_count"),
    "friends_count": S("data", "public_metrics", "following_count"),
    "lang": K(None),
    "listed_count": S("data", "public_metrics", "listed_count"),
    "favourites_count": K("0") >> F(int),
    "statuses_count": S("data", "public_metrics", "tweet_count"),
    "created_at": S("data", "created_at") >> F(v2_to_v1_date),
    "default_profile": K(True),
    "default_profile_image": If(
        S("data", "profile_image_url") == K(default_profile_image_url),
        K(True),
        K(False),
    ),
    "verified": S("data", "verified"),
    "profile_image_url_https": S("data", "profile_image_url"),
}


def bend_user(user):
    user = {"data": user}
    return bend(mapping=V2_TO_V1_USER, source=user)


V2_TO_V1_TWEET = {
    "created_at": S("data", "created_at") >> F(v2_to_v1_date),
    "id": S("data", "id") >> F(int),
    "id_str": S("data", "id"),
    "text": S("data", "text"),
    "entities": {
        "user_mentions": OptionalS("data", "entities", "mentions", default=[])
        >> ForallBend(
            {
                "screen_name": S("username"),
                "id": S("id") >> F(int),
                "id_str": S("id"),
                "indices": [S("start"), S("end")],
            }
        ),
        "symbols": OptionalS("data", "entities", "cashtags", default=[])
        >> ForallBend(
            {
                "text": S("tag"),
                "indices": [S("start"), S("end")],
            }
        ),
        "hashtags": OptionalS("data", "entities", "hashtags", default=[])
        >> ForallBend(
            {
                "text": S("tag"),
                "indices": [S("start"), S("end")],
            }
        ),
        "urls": OptionalS("data", "entities", "urls", default=[])
        >> ForallBend(
            {
                "url": S("url"),
                "expanded_url": Alternation(S("expanded_url"), S("url")),
                "display_url": Alternation(S("display_url"), S("url")),
                "indices": [S("start"), S("end")],
            }
        ),
    },
    "possibly_sensitive": S("data", "possibly_sensitive"),
    "lang": S("data", "lang"),
    "user": OptionalS("includes", "users")[0] >> F(bend_user),
    "retweet_count": S("data", "public_metrics", "retweet_count"),
    "favorite_count": S("data", "public_metrics", "like_count"),
}
