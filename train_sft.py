import argparse
import math
import torch

from dataset.sft_dataset import SupervisedDataset
from trainer.sft import SFTTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.trainer import get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logging.basicConfig(level=logging.INFO)

def train(args):
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain,
                                              trust_remote_code=True, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.pretrain,
                                                 trust_remote_code=True, cache_dir=args.cache)
    model.to(device)

    optim = Adam(model.parameters(), lr=args.lr)

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.dataset,
                                      max_len=args.max_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  pin_memory=True)

    step_per_epoch = len(train_dataloader)
    max_steps = math.ceil(args.max_epochs * step_per_epoch)
    lr_scheduler = get_scheduler('cosine', optim,
                                 num_warmup_steps=math.ceil(max_steps * 0.03),
                                 num_training_steps=max_steps)

    trainer = SFTTrainer(model=model, optim=optim, lr_scheduler=lr_scheduler,
                         max_epochs=args.max_epochs, device=device)
    trainer.fit(train_dataloader=train_dataloader, logger=logging)

    state = model.state_dict()
    torch.save(state, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='ds/alpaca-en.json')
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument('--save_path', type=str, default='sft_output')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()

    train(args)

