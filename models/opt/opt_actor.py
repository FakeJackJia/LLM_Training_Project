from typing import Optional
from transformers.models.opt import OPTForCausalLM, OPTConfig

from ..base.actor import Actor

class OPTActor(Actor):
    def __init__(self, pretrained: Optional[str] = None, config: Optional[OPTConfig] = None, cache_dir=None):
        if pretrained:
            model = OPTForCausalLM.from_pretrained(pretrained, cache_dir=cache_dir)
        elif config:
            model = OPTForCausalLM(config)
        else:
            model = OPTForCausalLM(OPTConfig())

        super(OPTActor, self).__init__(model)