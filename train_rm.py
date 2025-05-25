import argparse
import torch
from dataset.rm_dataset import RmDataset
from models.loss import LogSigClass
from models.opt.opt_rm import OPTRM
from trainer.rm import RewardModelTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.utils import jsonl_load
import logging
logging.basicConfig(level=logging.INFO)

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda'
    model = OPTRM(pretrained=args.pretrain, cache_dir=args.cache).to(device)
    if args.model_path:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    optim = Adam(model.parameters(), args.lr)
    loss_fn = LogSigClass()

    train_data = jsonl_load(args.dataset)
    logging.info("Tokenizing...")
    train_dataset = RmDataset(train_data, tokenizer, args.max_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  pin_memory=True)

    lr_scheduler = CosineAnnealingLR(optim, max(len(train_dataloader) // 100, 1))
    trainer = RewardModelTrainer(model=model, optim=optim, lr_scheduler=lr_scheduler,
                                 loss_fn=loss_fn, max_epochs=args.max_epochs, device=device)

    logging.info('Training start')
    trainer.fit(train_dataloader=train_dataloader)

    state_dict = model.state_dict()
    torch.save(state_dict, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ds/rm.jsonl')
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument('--save_path', type=str, default='rm_output')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()

    train(args)