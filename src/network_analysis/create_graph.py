from dotenv import load_dotenv
import os
from pymongo import MongoClient
import networkx as nx
from collections import Counter
from tqdm import tqdm

# load environment variables (like the Twitter API bearer token) from .env file
load_dotenv()

graph_dir = os.path.join(os.path.dirname(__file__), "graph_files")
subject_graph_file = os.path.join(graph_dir, "subjects.gml")
mentions_graph_file = os.path.join(graph_dir, "mentions.gml")
attributes_graph_file = os.path.join(graph_dir, "attributes.gml")

config = {
    "rerun": True,
    "min_subject_tweets": 500,
    "min_peer_tweets": 50,
    "top-k": 15,
}


def create_subjects_graph(database):
    # subjects collection
    subjects_collection = database.subjects_collection

    cursor = subjects_collection.find(
        {"timeline_tweets_count": {"$gte": config["min_subject_tweets"]}}
    )

    g = nx.DiGraph()

    for s in cursor:
        g.add_node(s["id"], username=s["username"])

    nx.write_gml(g, subject_graph_file)

    return g


def load_subjects_graph(database):
    # if os.path.exists(subject_graph_file) and not config["rerun"]:
    if os.path.exists(subject_graph_file):
        print("Reading subjects graph from file...")
        graph = nx.read_gml(subject_graph_file)
    else:
        print("Create subjects graph...")
        graph = create_subjects_graph(database)
    return graph


def add_mentions(database, graph):
    # timelines collection
    timelines_collection = database.timelines_collection
    user_timeline_no_RT_en = {
        "author_id": None,
        "referenced_tweets.0.type": {"$ne": "retweeted"},
        "lang": "en",
    }

    new_nodes = []
    new_edges = []
    for user_id in tqdm(graph.nodes, position=0):
        user_timeline_no_RT_en["author_id"] = user_id

        mentions = Counter()
        mentioned_usernames = {}

        t_cursor = timelines_collection.find(
            user_timeline_no_RT_en,
        ).limit(config["min_subject_tweets"])

        for t in t_cursor:
            if "entities" in t and "mentions" in t["entities"]:
                mentioned_ids = [mention["id"] for mention in t["entities"]["mentions"]]
                for mention in t["entities"]["mentions"]:
                    mentioned_usernames[mention["id"]] = mention["username"]
                mentions = mentions + Counter(mentioned_ids)
        t_cursor.close()

        # top-K mentioned users
        top_mentioned = mentions.most_common(config["top-k"])

        for mentioned_id, n_mentions in tqdm(top_mentioned, position=1, leave=False):
            # graph.add_node(mentioned_id, username=mentioned_usernames[mentioned_id])
            new_nodes.append(
                (mentioned_id, {"username": mentioned_usernames[mentioned_id]})
            )
            # graph.add_edge(user_id, mentioned_id, weight=n_mentions)
            new_edges.append((user_id, mentioned_id, {"weight": n_mentions}))

    # add new nodes and edges
    graph.add_nodes_from(new_nodes)
    graph.add_edges_from(new_edges)

    nx.write_gml(graph, mentions_graph_file)

    return graph


def load_mentions_graph(database, base_graph):
    # if os.path.exists(mentions_graph_file) and not config["rerun"]:
    if os.path.exists(mentions_graph_file):
        print("Reading mentions graph from file...")
        graph = nx.read_gml(mentions_graph_file)
    else:
        print("Adding mentioned users to graph...")
        graph = add_mentions(database, base_graph)
    return graph


def add_attributes(database, graph):
    # users collection
    users_collection = database.users_collection

    subjects_collection = database.subjects_collection
    peers_collection = database.peers_collection

    public_metrics = {}
    verified = {}
    type = {}
    enough_tweets = {}
    for user_id in tqdm(graph.nodes):
        user = users_collection.find_one({"id": user_id})
        if user is not None:
            public_metrics[user_id] = user["public_metrics"]
            verified[user_id] = user["verified"]

        subject = subjects_collection.find_one({"id": user_id})
        peer = peers_collection.find_one({"id": user_id})
        is_subject = bool(subject is not None)
        is_peer = bool(peer is not None)

        if is_subject and is_peer:
            type[user_id] = "SP"
            enough_tweets[user_id] = bool(
                subject["timeline_tweets_count"] >= config["min_subject_tweets"]
            )
        elif is_subject:
            type[user_id] = "S"
            enough_tweets[user_id] = bool(
                subject["timeline_tweets_count"] >= config["min_subject_tweets"]
            )
        elif is_peer:
            type[user_id] = "P"
            enough_tweets[user_id] = bool(
                peer["timeline_tweets_count"] >= config["min_peer_tweets"]
            )
        else:
            type[user_id] = "None"
            enough_tweets[user_id] = False

    nx.set_node_attributes(graph, public_metrics, name="public_metrics")
    nx.set_node_attributes(graph, verified, name="verified")
    nx.set_node_attributes(graph, type, name="type")
    nx.set_node_attributes(graph, enough_tweets, name="enough_tweets")

    nx.write_gml(graph, attributes_graph_file)

    return graph


def load_attributes_graph(database, base_graph):
    if os.path.exists(attributes_graph_file) and not config["rerun"]:
        print("Reading attributes graph from file...")
        graph = nx.read_gml(attributes_graph_file)
    else:
        print("Adding attributes to graph...")
        graph = add_attributes(database, base_graph)
    return graph


def main():
    mongo_conn = MongoClient(os.environ["MONGO_CONN"])
    db = mongo_conn.twitter  # our database

    graph = load_subjects_graph(database=db)

    graph = load_mentions_graph(database=db, base_graph=graph)

    graph = load_attributes_graph(database=db, base_graph=graph)


if __name__ == "__main__":
    main()
