import os, uuid
import optuna
from dataclasses import replace

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.config import finalize_args, setup_wandb
from algos.ppo import Args, train

RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4().hex[:8]))

if __name__ == "__main__":
    args = Args(
        env_id="CartPole-v1",
        wandb_project_name="Parallel_RL_Test",
        total_timesteps=500_000,
        eval_interval=50_000,
        eval_episodes=10,
        cuda=False,
        track=False,
    )

    # make exp_name correctly from your algo file, not run_dir
    args = finalize_args(args, "ppo.py")  # or __file__ from algos/ppo if you pass it through

    # run naming

    run_dir = os.path.join(os.path.dirname(__file__), "runs", f"TEST_TRIAL")
    os.makedirs(run_dir, exist_ok=True)

    setup_wandb(args, RUN_ID)  # update setup_wandb to use args.run_name (and group)


    train(args, run_dir=run_dir)
