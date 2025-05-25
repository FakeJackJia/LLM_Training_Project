import torch
import torch.nn as nn

# pairwise loss
class LogSigClass(nn.Module):
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = - log_probs.mean()

        return loss

class LMLoss(nn.Module):
    def __init__(self):
        super(LMLoss, self).__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # [batch_size*seq_len, vocab_size]
        # [batch_size*seq_len]
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

class PolicyLoss(nn.Module):
    def __init__(self, clip_eps: float = 0.2):
        super(PolicyLoss, self).__init__()

        self.clip_eps = clip_eps

    def forward(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor):
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        clip_loss = torch.min(surr1, surr2)

        return -clip_loss.mean()

class ValueLoss(nn.Module):
    def __init__(self):
        super(ValueLoss, self).__init__()

    def forward(self, values: torch.Tensor, reward: torch.Tensor):
        return ((values - reward) ** 2).mean()