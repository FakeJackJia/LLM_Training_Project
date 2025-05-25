from typing import Optional
import torch
import torch.nn as nn

from ..utils import masked_mean

class Critic(nn.Module):
    def __init__(self, model: nn.Module, value_head: nn.Module):
        super(Critic, self).__init__()

        self.model = model
        self.value_head = value_head

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # [batch, seq_len, d_dim]
        last_hidden_states = outputs['last_hidden_state']
        # [batch, seq_len, 1] -> [batch, seq_len]
        values = self.value_head(last_hidden_states).squeeze(-1)
        value = masked_mean(values, attention_mask, dim=1)

        return value