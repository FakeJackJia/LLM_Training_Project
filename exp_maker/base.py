from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from models.base.actor import Actor
from models.base.critic import Critic

@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.Tensor]

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.base_action_log_probs = self.base_action_log_probs.to(device)
        self.values = self.values.to(device)
        self.reward = self.reward.to(device)
        self.advantages = self.advantages.to(device)

        if self.attention_mask:
            self.attention_mask = self.attention_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.base_action_log_probs = self.base_action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.reward = self.reward.pin_memory()
        self.advantages = self.advantages.pin_memory()

        if self.attention_mask:
            self.attention_mask = self.attention_mask.pin_memory()

        return self

class ExpMaker(ABC):
    def __init__(self, actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: nn.Module,
                 kl_coef: float = 0.1):
        super(ExpMaker, self).__init__()

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs):
        raise NotImplementedError()