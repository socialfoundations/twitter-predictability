import re

# Preprocessing steps that can be used with Huggingface's Dataset's .map() function


def remove_urls(x):
    return re.sub(r"http\S+", "", x)


def remove_urls_batch(examples):
    res = []
    for e in examples["text"]:
        res.append(remove_urls(e))
    return {"text": res}


def remove_extra_spaces(x):
    return " ".join(x.split())


def remove_extra_spaces_batch(examples):
    res = []
    for e in examples["text"]:
        res.append(remove_extra_spaces(e))
    return {"text": res}


def replace_special_characters(x):
    x = x.replace("&amp;", "&")
    x = x.replace("&lt;", "<")
    x = x.replace("&gt;", ">")
    return x


def replace_special_characters_batch(examples):
    res = []
    for e in examples["text"]:
        res.append(replace_special_characters(e))
    return {"text": res}
