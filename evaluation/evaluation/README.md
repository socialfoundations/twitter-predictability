### Visualization

**Plot NLLs of subject's tweets**

Prints subject tweets with a heatmap-like background color. Color corresponds to magnitude of NLL.
```
python -m visualize plot_NLLs --subject_id=\"<subject_id>\"
```

Other options:
```
--context=<context_name>: If none, then plots NLLs calculated without any additional context. Options: [none, user, peer, random_user, random_tweet]. Default: none
--model=<model_name>: Which model's NLLs and tokenizer to use. Default: gpt2-xl
--token_level=<True/False>: Whether to plot word-level or token-level colorization. Default: False
```

**Plot improvement on NLLs of subject's tweets**

Prints subject tweets with a heatmap-like background color. Color corresponds to magnitude of improvement on NLL.
```
python -m visualize plot_improvement --subject_id=\"<subject_id>\"
```

Other options:
```
--base=<context_name>: Base context: plotted improvement is wrt. this context. Options: [none, user, peer, random_user, random_tweet]. Default: none
--context=<context_name>: What we compare to the base context. Default: user
--model=<model_name>: Which model's NLLs and tokenizer to use. Default: gpt2-xl
--token_level=<True/False>: Whether to plot word-level or token-level colorization. Default: False.
```