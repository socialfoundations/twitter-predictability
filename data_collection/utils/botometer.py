import logging, os
import requests

# main logger
main_logger = logging.getLogger("main")

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
