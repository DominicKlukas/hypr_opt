import optuna

from algos.sac_fetch_baseline import Args, train


def objective(trial: optuna.Trial) -> float:
    args = Args(
        env_id="FetchSlideDense-v4",
        total_timesteps=1_000_000,
        eval_interval=25_000,
        eval_episodes=25,
        cuda=False,
        track=False,
        policy_lr=trial.suggest_float("policy_lr", 1e-4, 1e-3, log=True),
        q_lr=trial.suggest_float("q_lr", 1e-4, 2e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        tau=trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        seed=1000 + trial.number,
    )
    args.run_name = f"sac_fetch_slide_dense_v4_baseline__t{trial.number:06d}"
    run_dir = "experiments/sac_fetch_slide_dense_v4_baseline/runs/" + args.run_name
    return train(args, run_dir=run_dir, trial=trial)
