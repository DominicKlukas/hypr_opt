import random
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim

from common.logging import Logger
from common.pruning import PruneReporter
from common.replay_buffer import ReplayBuffer
from networks.dqn_snake_equivariant import EquivariantSnakeQNetwork


@dataclass
class Args:
    exp_name: str = "dqn_snake_equivariant"
    run_name: str = "run"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    log_dir: str = "runs"

    total_timesteps: int = 8_000_000
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1_000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000
    train_frequency: int = 4

    eval_interval: int = 250_000
    eval_episodes: int = 50


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@torch.no_grad()
def evaluate_policy(q_network, eval_env, device, n_episodes: int) -> float:
    q_network.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(q_network(obs_t), dim=1).item())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    q_network.train()
    return float(np.mean(returns))


def train(
    args: Args,
    make_env: Callable[[], gym.Env],
    run_dir: str,
    trial: optuna.Trial | None = None,
) -> float:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger = Logger(args, run_dir)
    pr = PruneReporter(trial=trial, eval_interval_steps=args.eval_interval)

    envs = gym.vector.SyncVectorEnv([make_env])
    eval_env = make_env()

    q_network = EquivariantSnakeQNetwork(envs.single_action_space.n).to(device)
    target_network = EquivariantSnakeQNetwork(envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    best_eval = -float("inf")

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.as_tensor(obs, device=device, dtype=torch.float32))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rb.add(obs, next_obs.copy(), actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for p, tp in zip(q_network.parameters(), target_network.parameters()):
                    tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

            if global_step % 1000 == 0:
                logger.log_scalars({"td_loss": loss.item(), "epsilon": epsilon}, global_step, prefix="losses")

        if pr.should_eval(global_step) or (global_step + 1 >= args.total_timesteps):
            eval_return = evaluate_policy(q_network, eval_env, device, n_episodes=args.eval_episodes)
            logger.log_scalar("charts/eval_return", eval_return, global_step)
            best_eval = max(best_eval, eval_return)
            pr.report(eval_return, global_step)

    envs.close()
    eval_env.close()
    logger.close()
    return best_eval


def main():
    raise SystemExit(
        "This module expects a Snake env factory from the atari-snake-dqn branch code. "
        "Use it by importing train(...) from an experiment script."
    )


if __name__ == "__main__":
    main()
