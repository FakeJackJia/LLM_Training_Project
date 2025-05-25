import torch
import torch.nn.functional as F

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # [1, 1, 1, 0, 0, 0]
    tensor = tensor * mask
    s = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim) # number of not masked
    mean = s / (mask_sum + 1e-8)

    return mean

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))

    return log_probs_labels.squeeze(-1)

def cal_action_log_probs(output: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
    logits = output['logits']
    log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])

    return log_probs

def compute_approx_kl(log_probs: torch.Tensor, log_probs_old: torch.Tensor):
    log_ratio = log_probs - log_probs_old
    approx_kl = (log_ratio.exp() - 1) - log_ratio

    return approx_kl.mean(dim=1)

def compute_reward(r, kl_coef: float, log_probs: torch.Tensor, log_probs_old: torch.Tensor) -> torch.Tensor:
    if kl_coef <= 0:
        return r

    kl = compute_approx_kl(log_probs, log_probs_old)
    reward = r - kl_coef * kl

    return reward