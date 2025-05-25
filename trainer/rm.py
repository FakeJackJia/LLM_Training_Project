from typing import Callable
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .base import Trainer

class RewardModelTrainer(Trainer):
    def __init__(self, model, optim: Optimizer, lr_scheduler, loss_fn: Callable,
                 max_epochs: int = 1, device='cuda'):
        super(RewardModelTrainer, self).__init__(max_epochs, model, optim)

        self.loss_fn = loss_fn
        self.scheduler = lr_scheduler
        self.device = torch.device(device)

    def _before_fit(self, train_dataloader: DataLoader):
        self.train_dataloader = train_dataloader

    def _train(self, epoch):
        self.model.train()

        for idx, (chosen_ids, c_mask, reject_ids, r_mask) in enumerate(self.train_dataloader):
            chosen_ids = chosen_ids.squeeze(1).to(self.device)
            c_mask = c_mask.squeeze(1).to(self.device)
            chosen_reward = self.model(chosen_ids, attention_mask=c_mask)

            reject_ids = reject_ids.squeeze(1).to(self.device)
            r_mask = r_mask.squeeze(1).to(self.device)
            reject_reward = self.model(reject_ids, attention_mask=r_mask)

            loss = self.loss_fn(chosen_reward, reject_reward)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if idx % 100 == 0:
                self.scheduler.step()