# Evaluation

# Dataset building
`evaluation/build_dataset.py` where we load the following:
- user_eval (250 most recent tweets of subject)
- user_context
- peer_context
- random_context (control - random tweets)

# Prompting
Main experiment for getting the conditional negative log-likelihood for each eval tweet of a single user: `evaluation/prompting.py`.