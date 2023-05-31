import unittest
import os
from data_collection.utils.converter import V2_TO_V1_TWEET, V2_TO_V1_USER
from jsonbender import bend
import json
from jsoncomparison import Compare, NO_DIFF

FILE_PATH = os.path.dirname(__file__)
TEST_FILES_PATH = os.path.join(FILE_PATH, "converter_examples")
DIFFS_PATH = os.path.join(FILE_PATH, "converter_diffs")

if not os.path.exists(DIFFS_PATH):
    os.makedirs(DIFFS_PATH)

compare_config = {
    "output": {
        "console": False,
        "file": {
            "allow_nan": True,
            "ensure_ascii": True,
            "indent": 4,
            "name": os.path.join(FILE_PATH, "difference.json"),
            "skipkeys": True,
        },
    }
}

# ignore these keys when comparing
user_exceptions = {
    "id": "*",  # V1 stored a 53-bit int representation of the 64-bit id, which can cause a mismatch between v2_id and v1_id
    "favourites_count": "*",  # because of default value
    "profile_location": "*",
    "entities": {"description": "*"},
    "utc_offset": "*",
    "time_zone": "*",
    "geo_enabled": "*",
    "status": "*",
    "contributors_enabled": "*",
    "is_translator": "*",
    "is_translation_enabled": "*",
    "profile_background_color": "*",
    "profile_background_image_url": "*",
    "profile_background_image_url_https": "*",
    "profile_background_tile": "*",
    "profile_image_url": "*",
    "profile_banner_url": "*",
    "profile_link_color": "*",
    "profile_sidebar_border_color": "*",
    "profile_sidebar_fill_color": "*",
    "profile_text_color": "*",
    "profile_use_background_image": "*",
    "has_extended_profile": "*",
    "default_profile": "*",  # because of default value
    "following": "*",
    "follow_request_sent": "*",
    "notifications": "*",
    "translator_type": "*",
    "withheld_in_countries": "*",
}

tweet_exceptions = {
    "id": "*",  # V1 stored a 53-bit int representation of the 64-bit id, which can cause a mismatch between v2_id and v1_id
    "truncated": "*",
    "entities": {"user_mentions": {"_list": {"name": "*", "id": "*"}}},
    "source": "*",
    "in_reply_to_status_id": "*",
    "in_reply_to_status_id_str": "*",
    "in_reply_to_user_id": "*",
    "in_reply_to_user_id_str": "*",
    "in_reply_to_screen_name": "*",
    "user": user_exceptions,
    "geo": "*",
    "coordinates": "*",
    "place": "*",
    "contributors": "*",
    "is_quote_status": "*",
    "quoted_status_id": "*",
    "quoted_status_id_str": "*",
    "quoted_status": "*",
    "favorited": "*",
    "retweeted": "*",
    "possibly_sensitive_appealable": "*",
}


class TestUserConverters(unittest.TestCase):
    def setUp(self):
        self.compare_config = compare_config
        self.compare_config["output"]["file"]["name"] = os.path.join(
            DIFFS_PATH, self.id() + ".json"
        )

    def _get_expected_actual(self, file1, file2):
        with open(os.path.join(TEST_FILES_PATH, file1)) as file:
            v2_user = json.load(file)
        v1_user_actual = bend(V2_TO_V1_USER, v2_user)
        with open(os.path.join(TEST_FILES_PATH, file2)) as file:
            v1_user_expected = json.load(file)
        return v1_user_expected, v1_user_actual

    def test_V2_to_V1_user_example(self):
        expected, actual = self._get_expected_actual("V2_user.json", "V1_user.json")
        diff = Compare(config=self.compare_config, rules=user_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)

    def test_V2_to_V1_user_example_1(self):
        expected, actual = self._get_expected_actual("V2_user_1.json", "V1_user_1.json")
        diff = Compare(config=self.compare_config, rules=user_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)

    def test_V2_to_V1_user_example_2(self):
        "Example with 'entities.url.urls.expanded_url' missing"
        expected, actual = self._get_expected_actual("V2_user_2.json", "V1_user_2.json")
        diff = Compare(config=self.compare_config, rules=user_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)


class TestTweetConverters(unittest.TestCase):
    def setUp(self):
        self.compare_config = compare_config
        self.compare_config["output"]["file"]["name"] = os.path.join(
            DIFFS_PATH, self.id() + ".json"
        )

    def _get_expected_actual(self, file1, file2):
        with open(os.path.join(TEST_FILES_PATH, file1)) as file:
            v2_user = json.load(file)
        v1_user_actual = bend(V2_TO_V1_TWEET, v2_user)
        with open(os.path.join(TEST_FILES_PATH, file2)) as file:
            v1_user_expected = json.load(file)
        return v1_user_expected, v1_user_actual

    def test_V2_to_V1_tweet_example(self):
        expected, actual = self._get_expected_actual("V2_tweet.json", "V1_tweet.json")
        diff = Compare(config=self.compare_config, rules=tweet_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)

    def test_V2_to_V1_tweet_example_1(self):
        expected, actual = self._get_expected_actual(
            "V2_tweet_1.json", "V1_tweet_1.json"
        )
        diff = Compare(config=self.compare_config, rules=tweet_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)

    def test_V2_to_V1_tweet_example_2(self):
        expected, actual = self._get_expected_actual(
            "V2_tweet_2.json", "V1_tweet_2.json"
        )
        diff = Compare(config=self.compare_config, rules=tweet_exceptions).check(
            expected, actual
        )
        self.assertEqual(diff, NO_DIFF)


if __name__ == "__main__":
    unittest.main()
