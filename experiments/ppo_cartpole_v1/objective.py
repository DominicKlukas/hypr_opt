import os, uuid
import optuna
from dataclasses import replace

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.config import finalize_args, setup_wandb
from algos.ppo import Args, train

RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4().hex[:8]))

def objective(trial: optuna.Trial) -> float:
    # suggest a couple params
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    ent = trial.suggest_float("ent_coef", 0.0, 0.05)

    args = Args(
        env_id="CartPole-v1",
        wandb_project_name="Parallel_RL_Test",
        total_timesteps=200_000,
        eval_interval=20_000,
        eval_episodes=20,
        cuda=False,
        track=True,
    )
    args = replace(args, learning_rate=lr, ent_coef=ent, seed=1000 + trial.number)
    run_dir = os.path.join(os.path.dirname(__file__), "runs", f"trial_{trial.number:06d}__{RUN_ID}")
    os.makedirs(run_dir, exist_ok=True)
    args = finalize_args(args, run_dir)
    setup_wandb(args)

    return train(args, run_dir=run_dir, trial=trial)
