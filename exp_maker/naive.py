import torch
from models.generate import generate_with_actor
from models.utils import cal_action_log_probs, compute_reward
from .base import ExpMaker, Experience

class NaiveExpMaker(ExpMaker):
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs):
        self.actor.train()
        self.critic.train()
        self.initial_model.eval()
        self.reward_model.eval()

        sequences, attention_mask = generate_with_actor(self.actor, input_ids, **generate_kwargs)

        actor_output = self.actor(sequences, attention_mask)
        action_log_probs = cal_action_log_probs(actor_output, sequences)
        base_model_output = self.initial_model(sequences, attention_mask)
        base_action_log_probs = cal_action_log_probs(base_model_output, sequences)

        value = self.critic(sequences, attention_mask)
        r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs)

        # r + V(next) - V(current) -> no next so -> r - V(current)
        advantage = reward - value
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, base_action_log_probs, value, reward, advantage, attention_mask)