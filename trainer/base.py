from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map

def to_device(x: Any, device: torch.device):
    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)

        return t

    return tree_map(_to, x)

class Trainer(ABC):
    def __init__(self, max_epochs: int, model: nn.Module, optimizer: Optimizer):
        self.max_epochs = max_epochs
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def _train(self, epoch):
        raise NotImplementedError()

    @abstractmethod
    def _before_fit(self, *args, **kwargs):
        raise NotImplementedError()

    def fit(self, *args, **kwargs):
        self._before_fit(*args, **kwargs)

        for epoch in tqdm(range(self.max_epochs)):
            self._train(epoch)

class OnPolicyTrainer(ABC):
    def __init__(self):
        super(OnPolicyTrainer, self).__init__()

    @abstractmethod
    def _learn(self):
        raise NotImplementedError()

    def fit(self, prompt_dataloader: DataLoader,
            pretrain_dataloader: DataLoader,
            num_episodes: int):

        self.prompt_dataloader = prompt_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.pretrain_iter = iter(self.pretrain_dataloader)

        for _ in tqdm(range(num_episodes)):
            self._learn()