from typing import Optional
import torch.nn as nn
from transformers.models.opt import OPTConfig, OPTModel

from ..base.critic import Critic

class OPTCritic(Critic):
    def __init__(self, pretrained: Optional[str] = None, config: Optional[OPTConfig] = None, cache_dir=None):
        if pretrained:
            model = OPTModel.from_pretrained(pretrained, cache_dir=cache_dir)
        elif config:
            model = OPTModel(config)
        else:
            model = OPTModel(OPTConfig())

        value_head = nn.Linear(model.config.word_embed_proj_dim, 1)
        super(OPTCritic, self).__init__(model, value_head)