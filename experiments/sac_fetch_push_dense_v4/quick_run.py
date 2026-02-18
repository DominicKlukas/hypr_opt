import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from algos.sac_fetch_equivariant import Args, train


if __name__ == "__main__":
    args = Args(env_id="FetchPushDense-v4", num_envs=1, total_timesteps=400, learning_starts=50, eval_interval=200, eval_episodes=1, dihedral_n=4, reg_rep_n=8, track=False, vector_env_mode="sync", cuda=False)
    args.run_name = "quick_run"
    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(train(args, run_dir=run_dir, trial=None))
