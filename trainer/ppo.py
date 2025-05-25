from typing import Dict
import torch
import torch.nn as nn
from exp_maker.naive import NaiveExpMaker, Experience
from models.base import Actor, Critic
from models.loss import LMLoss, PolicyLoss, ValueLoss
from torch.optim import Optimizer
from .base import OnPolicyTrainer, to_device

class PPOTrainer(OnPolicyTrainer):
    def __init__(self, actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: nn.Module,
                 action_optim: Optimizer,
                 critic_optim: Optimizer,
                 kl_coef: float = 0.1,
                 c1: float = 0.5,
                 c2: float = 0.9,
                 eps_clip: float = 0.2,
                 device='cuda',
                 **generate_kwargs):
        super(PPOTrainer, self).__init__()

        self.generate_kwargs = generate_kwargs
        self.experience_maker = NaiveExpMaker(actor, critic, reward_model, initial_model, kl_coef)
        self.actor = actor
        self.critic = critic
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss()
        self.reg_loss_fn = LMLoss()
        self.critic_coef = c1
        self.reg_coef = c2
        self.actor_optim = action_optim
        self.critic_optim = critic_optim
        self.device = torch.device(device)

    def _training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        actor_loss = self.actor_loss_fn(experience.action_log_probs, experience.base_action_log_probs, experience.advantages)
        critic_loss = self.critic_loss_fn(experience.values, experience.reward)
        batch = next(self.pretrain_iter)
        batch = to_device(batch, self.device)
        reg_log_probs = self.actor(batch['input_ids'])['logits']
        reg_loss = self.reg_loss_fn(reg_log_probs, batch['labels'])

        total_loss = actor_loss + self.critic_coef * critic_loss + self.reg_coef * reg_loss
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

        return {'reward': experience.reward.mean().item()}

    def _learn(self):
        for idx, prompts in enumerate(self.prompt_dataloader):
            experience = self.experience_maker.make_experience(**prompts, **self.generate_kwargs)
            to_device(experience, self.device)
            self._training_step(experience)