import os
import random
import time
from dataclasses import dataclass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 - needed to register Fetch envs
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro

from common.config import setup_wandb
from common.fetch_obs import FetchObsWrapper
from common.logging import Logger
from common.pruning import PruneReporter
from common.replay_buffer import ReplayBuffer
from networks.sac_equivariant import EquivariantActor, EquivariantSoftQNetwork


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


@dataclass
class Args:
    exp_name: str = "sac_fetch_equivariant"
    run_name: str = "run"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "hypr_opt_fetch"
    wandb_entity: str | None = None
    capture_video: bool = False
    log_dir: str = "runs"

    env_id: str = "FetchReachDense-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    dihedral_n: int = 16
    reg_rep_n: int = 64

    eval_interval: int = 25_000
    eval_episodes: int = 25
    vector_env_mode: str = "async"
    stdout_log_interval: int = 1_000


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FetchObsWrapper(env, env_id)
        env.action_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def evaluate_policy(actor, eval_env, device, n_episodes: int, base_seed: int) -> float:
    actor.eval()
    returns = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, mean_action = actor.get_action(obs_t)
            action = mean_action.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    actor.train()
    return float(np.mean(returns))


def train(args: Args, run_dir: str, trial: optuna.Trial | None = None) -> float:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available and args.cuda else "cpu")
    if args.cuda and not cuda_available:
        print("[warn] cuda=True requested but torch.cuda.is_available() is False; falling back to CPU.", flush=True)
    print(
        (
            f"[start] env={args.env_id} device={device} total_timesteps={args.total_timesteps} "
            f"num_envs={args.num_envs} dihedral_n={args.dihedral_n} reg_rep_n={args.reg_rep_n} "
            f"batch_size={args.batch_size} eval_interval={args.eval_interval}"
        ),
        flush=True,
    )

    if args.track:
        setup_wandb(args, run_id=args.run_name, trial=trial)

    logger = Logger(args, run_dir)
    pr = PruneReporter(trial=trial, eval_interval_steps=args.eval_interval)

    env_fns = [make_env(args.env_id, args.seed + i, i, args.capture_video, args.run_name) for i in range(args.num_envs)]
    if args.vector_env_mode == "sync":
        envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        envs = gym.vector.AsyncVectorEnv(env_fns, daemon=True, shared_memory=False)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "SAC requires continuous actions"

    eval_env = make_env(args.env_id, args.seed + 10_000, 0, False, args.run_name)()

    actor = EquivariantActor(envs, args.dihedral_n, args.reg_rep_n).to(device)
    qf1 = EquivariantSoftQNetwork(envs, args.dihedral_n, args.reg_rep_n).to(device)
    qf2 = EquivariantSoftQNetwork(envs, args.dihedral_n, args.reg_rep_n).to(device)
    qf1_target = EquivariantSoftQNetwork(envs, args.dihedral_n, args.reg_rep_n).to(device)
    qf2_target = EquivariantSoftQNetwork(envs, args.dihedral_n, args.reg_rep_n).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape, device=device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    best_eval = -float("inf")

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rb.add(obs, next_obs.copy(), actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next = qf1_target(data.next_observations, next_actions)
                qf2_next = qf2_target(data.next_observations, next_actions)
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next.view(-1)

            qf1_values = qf1(data.observations, data.actions).view(-1)
            qf2_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                sps = int(global_step / (time.time() - start_time)) if global_step > 0 else 0
                logger.log_scalars(
                    {
                        "qf1_loss": qf1_loss.item(),
                        "qf2_loss": qf2_loss.item(),
                        "qf_loss": qf_loss.item() / 2.0,
                        "qf1_values": qf1_values.mean().item(),
                        "qf2_values": qf2_values.mean().item(),
                        "alpha": alpha,
                        "sps": sps,
                    },
                    global_step,
                    prefix="losses",
                )
                if args.stdout_log_interval > 0 and global_step % args.stdout_log_interval == 0:
                    print(
                        (
                            f"[train] step={global_step} sps={sps} qf_loss={(qf_loss.item() / 2.0):.4f} "
                            f"alpha={alpha:.4f} device={device}"
                        ),
                        flush=True,
                    )

        if pr.should_eval(global_step) or (global_step + 1 >= args.total_timesteps):
            eval_return = evaluate_policy(
                actor,
                eval_env,
                device,
                n_episodes=args.eval_episodes,
                base_seed=args.seed + 100_000 + (global_step // max(args.eval_interval, 1)) * 1_000,
            )
            logger.log_scalar("charts/eval_return", eval_return, global_step)
            best_eval = max(best_eval, eval_return)
            pr.report(eval_return, global_step)
            print(
                f"[eval] step={global_step} eval_return={eval_return:.4f} best_eval={best_eval:.4f}",
                flush=True,
            )

    envs.close()
    eval_env.close()
    logger.close()

    if args.track:
        import wandb

        wandb.finish()

    print(f"[done] best_eval={best_eval:.4f}", flush=True)
    return best_eval


def main():
    args = tyro.cli(Args)
    if args.run_name == "run":
        args.run_name = f"{args.env_id}__{args.exp_name}__seed{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.log_dir, args.exp_name, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    score = train(args, run_dir=run_dir, trial=None)
    print(f"best_eval_return={score}", flush=True)


if __name__ == "__main__":
    main()
