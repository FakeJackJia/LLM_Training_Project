from typing import Any, Optional, Callable, Dict
import torch
import torch.nn as nn
from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

def prepare_logits_processor(top_k: Optional[int] = None, top_p: Optional[float] = None,
                             temperature: Optional[float] = None) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()

    if temperature and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))

    return processor_list

def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    return unfinished_sequences.max() == 0

def generate(model: nn.Module,
             input_ids: torch.Tensor,
             max_length: int,
             eos_token_id: int,
             pad_token_id: int,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], Dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[Dict, Any], Dict]] = None,
             **model_kwargs) -> torch.Tensor:
    if input_ids.size(1) >= max_length:
        return input_ids

    logits_processor = prepare_logits_processor(top_k, top_p, temperature)
    # [batch]
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    for _ in range(input_ids.size(1), max_length):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn else {'input_ids': input_ids}
        outputs = model(**model_inputs)

        next_token_logits = outputs['logits'][:, -1, :]
        next_token_logits = logits_processor(input_ids, next_token_logits)
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        if update_model_kwargs_fn:
            model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)

        unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

    return input_ids

@torch.no_grad()
def generate_with_actor(actor_model: nn.Module, input_ids: torch.Tensor, **kwargs):
    sequence = generate(actor_model, input_ids, **kwargs)
    attention_mask = None
    pad_token_id = kwargs.get('pad_token_id', None)
    if pad_token_id:
        attention_mask = sequence.not_equal(pad_token_id).to(dtype=torch.long, device=sequence.device)

    return sequence, attention_mask