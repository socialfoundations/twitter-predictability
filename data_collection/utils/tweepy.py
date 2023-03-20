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
