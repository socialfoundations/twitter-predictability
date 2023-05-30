from prompting import user_nlls, torch_compute_confidence_interval
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

config = {
    "device": "cpu",
    "num_users": 15,
    "model_id": "gpt2",
}


def main():
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database

    subjects_collection = db["subjects_collection"]
    subjects = subjects_collection.aggregate(
        [{"$sample": {"size": config["num_users"]}}]
    )

    nlls_config = {
        "device": config["device"],
        "model_id": config["model_id"],
        "ctxt_len": 900,
        "seq_sep": "\n",
        "batched": True,
        "batch_size": 2,
        "token_level_nlls": True,
    }

    modes = ["none", "user", "peer", "random"]

    results = []

    for s in subjects:
        nlls_config["user_id"] = s["id"]
        for m in modes:
            print(f"Subject id: {s['id']} / Mode: {m} ")
            nlls_config["mode"] = m
            try:
                nlls = user_nlls(config=nlls_config)
                mean, ci = torch_compute_confidence_interval(nlls, confidence=0.9)
                print(f"Negative log-likelihood (mean): {mean:.4f} +/- {ci:.4f}")
                results.append({"id": s["id"], "mode": m, "mean": mean, "ci": ci})
            except:
                break

    df = pd.DataFrame(results)
    df.to_csv("results/rand_users_nll.csv", index=False)


if __name__ == "__main__":
    main()
