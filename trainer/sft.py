from typing import Optional
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .base import Trainer, to_device

class SFTTrainer(Trainer):
    def __init__(self, model, optim: Optimizer, lr_scheduler, max_epochs: int = 2,
                 batch_size: int = 2, device='cuda'):
        super(SFTTrainer, self).__init__(max_epochs, model, optim)
        self.scheduler = lr_scheduler
        self.batch_size = batch_size
        self.device = torch.device(device)

    def _before_fit(self, train_dataloader: DataLoader, logger: Optional = None):
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.total_loss = 0

    def _train(self, epoch):
        self.model.train()

        for batch_id, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            outputs = self.model(batch['input_ids'],
                                 labels=batch['labels'])
            loss = outputs.loss
            self.total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.logger.info({'loss': self.total_loss,
                              'lr': self.scheduler.get_last_lr()[0],
                              'epoch': epoch,
                              'batch_id': batch_id
                              })