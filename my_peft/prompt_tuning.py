import torch
import torch.nn as nn
import math

class PromptEmbedding(nn.Module):
    def __init__(self, config, word_embedding):
        super(PromptEmbedding, self).__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = nn.Embedding(total_virtual_tokens, config.token_dim)

        if config.prompt_tuning_init == 'text':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.tokenzier_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)['input_ids']
            num_text_tokens = len(init_token_ids)

            if num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps

            init_token_ids = init_token_ids[:total_virtual_tokens]
            word_embeds = word_embedding(torch.LongTensor(init_token_ids)).detach().clone()
            word_embeds = word_embeds.to(torch.float32)

            self.embedding.weight = nn.Parameter(word_embeds)

    # [0, 1, 2, 3, 4, 5]
    def forward(self, indices):
        prompt_embs = self.embedding(indices)

        return prompt_embs