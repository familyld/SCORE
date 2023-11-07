# This is used to run SCORE on D4RL-MuJoCo tasks

import os
import argparse
import numpy as np
import torch
import gym, d4rl

import score.SCORE
import score.utils as utils

from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from utils import *

# Training
def train_SCORE(device, args):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    set_seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": device,
        "discount": args.discount,
        "tau": args.tau,
        "lr": args.lr,
        # TD3
        "expl_noise": args.expl_noise,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # BC
        "alpha": args.alpha,
        # SCORE
        "num_ensemble": args.num_ensemble,
        "beta": args.beta,
        "bc_decay": args.bc_decay,
        "spectral_norm": args.spectral_norm,
    }

    # Initialize policy
    policy = score.SCORE.SCORE(**kwargs)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, args.num_ensemble, ber_mean=args.ber_mean)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    replay_buffer.set_mask()
    mean,std = replay_buffer.normalize_states() 
    replay_buffer.normalize_rewards()
    print('Loaded buffer')
       
    snapshot = policy.get_snapshot()
    
    for epoch in gt.timed_for(
            range(int(args.epochs)),
            save_itrs=True,
    ):
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        gt.stamp('training', unique=False)
        paths = eval_policy(policy, env, mean, std, max_path_length=args.max_path_length)
        gt.stamp('evaluation sampling')
        log_stats(epoch, policy, paths)
        policy.end_epoch(epoch)

    snapshot = policy.get_snapshot()
    logger.save_itr_params(epoch, snapshot)
    gt.stamp('saving params') 

# Evaluation
def eval_policy(policy, eval_env, mean, std, max_path_length=1000, eval_episodes=10):
    paths = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        observations, actions, rewards, terminals = [], [], [], []
        path_length = 0

        while path_length < max_path_length:
            s = (np.array(state).reshape(1,-1) - mean)/std
            a = policy.select_action(s)
            next_o, r, d, _ = eval_env.step(a)
            observations.append(state)
            actions.append(a)
            terminals.append(d)
            rewards.append(r)
            path_length += 1
            if d:
                break
            state = next_o
        
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        path = dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
        )
        paths.append(path)

    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-random-v0")               # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)                               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")                           # Prepends name to filename
    parser.add_argument("--eval_freq", default=1e3, type=int)                        # How often (time steps) we evaluate
    parser.add_argument("--epochs", default=1e3, type=int)                           # Maximum epoch
    parser.add_argument("--batch_size", default=100, type=int)                       # Mini batch size for networks
    parser.add_argument("--discount", default=0.99, type=float)                      # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                          # Target network update rate
    parser.add_argument("--phi", default=0.05, type=float)                           # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--output_dir", default="output", type=str)                  # Ouput dir
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--max_path_length", default=1000, type=int)
    parser.add_argument('--version', default="normal", type=str)
    parser.add_argument("--gpu", default="", type=str)

    # TD3
    parser.add_argument("--expl_noise", default=1.0, type=float)                     # Std of Gaussian exploration noise
    parser.add_argument("--policy_noise", default=0.2, type=float)                   # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                     # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                        # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5, type=float)                          # BC coefficient
    # SCORE
    parser.add_argument('--num_ensemble', default=5, type=int)                       # Number of ensemble networks
    parser.add_argument('--beta', default=0.2, type=float)                           # Penalty coefficient
    parser.add_argument('--bc_decay', default=0.98, type=float)                      # Decay ratio
    parser.add_argument('--ber_mean', default=1.0, type=float)                       # Mask ratio for bootstrapped sampling
    parser.add_argument("--spectral_norm", action='store_true', default=False)
    args = parser.parse_args()    

    file_name = f"SCORE_{args.env_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Setting: Training SCORE, Env: {args.env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(f"./output/{args.env_name}"):
        os.makedirs(f"./output/{args.env_name}")
    setting = f"{args.env_name}_{args.seed}"
    setup_logger(f"SCORE({args.version})_{setting}", variant=vars(args), base_log_dir=f"./output/{args.env_name}")

    if (args.gpu != ""):
        enable_gpus(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_SCORE(device, args)
