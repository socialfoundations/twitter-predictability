import re

# Preprocessing steps that can be used with Huggingface's Dataset's .map() function


def remove_urls(x):
    return {"text": re.sub(r"http\S+", "", x["text"])}


def remove_extra_spaces(x):
    return {"text": " ".join(x["text"].split())}


def replace_special_characters(x):
    text = x["text"]
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    return {"text": text}


def remove_mentions(x):
    return {"text": re.sub(r"@\S+", "", x["text"])}


def remove_hashtags(x):
    return {"text": re.sub(r"#\S+", "", x["text"])}
