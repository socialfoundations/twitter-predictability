# Tweet Predictability
Code repository of arxiv preprint "**Limits to Predicting Online Speech Using Large Language Models**" (https://arxiv.org/abs/2407.12850). We study the predictability of online speech on Twitter. The significance of studying predictability is far-reaching; it helps us frame questions such as social influence, information diffusion and predicting sensitive author information.

Using **6.25M tweets** from **>5000 users** as the base of our study and with **language models of up to 70B parameters**, we find that **users' own history is most predictive of their future posts**.
We contrast this with posts from their social circle, and find that they consistently contain less predictive information. This result replicates across models and experimental methods (in-context learning as well as finetuning).

We additionally find that **the extent to which we can predict online speech is limited** even with state-of-the-art language models. Our observations do not suggest that peers exert an outsize influence on an individual's online posts. Concerns that large language models have made our individual expression predictable are not supported by our findings. 

## Data collection
Under `data_collection/`.

## Prompting experiments
Under `evaluation/`.

## Finetuning experiments
Under `finetune/`.

## Cite
```
@misc{remeli2024limitspredictingonlinespeech,
      title={Limits to Predicting Online Speech Using Large Language Models}, 
      author={Mina Remeli and Moritz Hardt and Robert C. Williamson},
      year={2024},
      eprint={2407.12850},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12850}, 
}
```