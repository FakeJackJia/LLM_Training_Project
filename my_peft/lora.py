import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LoraLayer:
    def __init__(self, r: int, lora_alpha: int, lora_drop_out: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha

        if lora_drop_out > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_drop_out)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapter = False

class Linear(nn.Linear, LoraLayer):
    def __init__(self, in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 merge_weights: bool = True,
                 **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        if r > 0:
            # [r, in]
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # [out, r]
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

        if not mode and self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling

            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= self.lora_B.weight @ self.lora_A.weight * self.scaling

            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x):
        if self.disable_adapter:
            if self.r > 0 and self.merged:
                self.weight.data -= self.lora_B.weight @ self.lora_A.weight * self.scaling
                self.merged= False

            return F.linear(x, self.weight, bias=self.bias)

        elif self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            return result + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        return F.linear(x, self.weight, bias=self.bias)

class MergedLinear(nn.Linear, LoraLayer):
    # q, k, v = xW
    # [True, False, True] for enable_lora
    def __init__(self, in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 enable_lora: List[bool] = [False],
                 merge_weights: bool = True,
                 **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        self.enable_lora = enable_lora
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_index = self.weight.new_zeros((out_features), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_index[enable_lora, :] = True
            self.lora_index = self.lora_index.view(-1)

    def zero_pad(self, x):
        # size of x: [..., out / 3 * 2]
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_index] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))

        return result.view((*x.shape[:-1], self.out_features))

    def delta_w_cal(self):
        delta_w = F.conv1d(
            self.lora_A.weight.data.unsqueeze(0),  # [batch, r*2, in]
            self.lora_B.weight.data.unsqueeze(-1),  # [out/3*2, r*2, 1]
            groups=sum(self.enable_lora)
        ).squeeze(0)  # [out/3*2, in]

        return delta_w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

        if not mode and self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                self.weight.data += torch.transpose(self.zero_pad(torch.transpos(self.delta_w_cal(), 0, 1) * self.scaling),0,1)
            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0 and any(self.enable_lora):
                self.weight.data -= torch.transpose(self.zero_pad(torch.transpos(self.delta_w_cal(), 0, 1) * self.scaling),0,1)
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x):
        if self.disable_adapter:
            if self.r > 0 and self.merged and any(self.enable_lora):
                self.weight.data -= torch.transpose(self.zero_pad(torch.transpos(self.delta_w_cal(), 0, 1) * self.scaling),0,1)
                self.merged = False

            return F.linear(x, self.weight, bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            # [r * seq_len]
            res_A = torch.transpose(self.lora_A(self.lora_dropout(x)), -2, -1)
            # [seq_len, out/3*2]
            res_B = torch.transpose(self.lora_B(res_A), -2, -1)

            return result + self.zero_pad(res_B)

        return F.linear(x, self.weight, bias=self.bias)