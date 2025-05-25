import argparse
import torch
from dataset.prompt_dataset import PromptDataset
from dataset.sft_dataset import SupervisedDataset
from models.opt import OPTRM, OPTCritic, OPTActor
from trainer.ppo import PPOTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logging.basicConfig(level=logging.INFO)

def train(args):
    logging.info('Preparing Models...')
    device = 'cuda'
    initial_model = AutoModelForCausalLM.from_pretrained(args.pretrain, cache_dir=args.cache).to(device)
    actor = OPTActor(pretrained=args.pretrain, cache_dir=args.cache).to(device)
    reward_model = OPTRM(pretrained=args.pretrain, cache_dir=args.cache).to(device)
    critic = OPTCritic(pretrained=args.pretrain, cache_dir=args.cache).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token

    actor_optim = Adam(actor.parameters(), lr=1e-7)
    critic_optim = Adam(critic.parameters(), lr=1e-7)

    logging.info('Preparing Datasets...')
    prompt_dataset = PromptDataset(data_path=args.prompt_dataset, tokenizer=tokenizer, device=args.device)
    prompt_dataloader = DataLoader(prompt_dataset, shuffle=True, batch_size=args.exp_batch_size)

    pretrain_dataset = SupervisedDataset(data_path=args.pretrain_dataset, tokenizer=tokenizer)
    pretrain_dataloader = DataLoader(pretrain_dataset, shuffle=True, batch_size=args.reg_batch_size)

    logging.info('Training Starts...')
    trainer = PPOTrainer(actor, critic, reward_model, initial_model, actor_optim, critic_optim,
                         kl_coef=args.kl_coef, c1=args.c1, c2=args.c2,eps_clip=args.eps_clip,
                         max_length=args.max_seq_len, eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id, top_k=50, temperature=1.0)

    trainer.fit(prompt_dataloader, pretrain_dataloader, num_episodes=args.num_episodes)

    state_dict = actor.state_dict()
    torch.save(state_dict, args.actor_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument('--prompt_dataset', type=str, default='ds/test.jsonl')
    parser.add_argument('--pretrain_dataset', type=str, default='ds/test.jsonl')
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--reg_batch_size', type=int, default=1)
    parser.add_argument('--exp_batch_size', type=int, default=4)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--c1', type=float, default=0.5)
    parser.add_argument('--c2', type=float, default=0.9)
    parser.add_argument('--max_seq_len', type=int, default=128)

    args = parser.parse_args()
    train(args)