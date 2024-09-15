from typing import Optional
import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import Trainer


class NoShuffleTrainer(Trainer):
    """
    A Trainer that doesn't shuffle data, but loops over it sequentially.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(eval_dataset)
