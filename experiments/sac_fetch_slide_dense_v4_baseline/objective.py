import os

import optuna

from algos.sac_fetch_baseline import Args, train


def objective(trial: optuna.Trial) -> float:
    # Narrowed search region around strong v2 baseline results:
    # policy_lr~2.44e-4, q_lr~3.11e-4, gamma~0.9878, tau~0.00886, batch=256
    mode = os.environ.get("BASELINE_SWEEP_MODE", "full").strip().lower()
    if mode == "narrow_v3":
        policy_lr = trial.suggest_float("policy_lr", 1.5e-4, 5e-4, log=True)
        q_lr = trial.suggest_float("q_lr", 2e-4, 8e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.982, 0.993)
        tau = trial.suggest_float("tau", 4e-3, 1.5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
    else:
        policy_lr = trial.suggest_float("policy_lr", 1e-4, 1e-3, log=True)
        q_lr = trial.suggest_float("q_lr", 1e-4, 2e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        tau = trial.suggest_float("tau", 1e-3, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    args = Args(
        env_id="FetchSlideDense-v4",
        total_timesteps=1_000_000,
        eval_interval=25_000,
        eval_episodes=25,
        cuda=True,
        track=False,
        policy_lr=policy_lr,
        q_lr=q_lr,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        seed=1000 + trial.number,
    )
    args.run_name = f"sac_fetch_slide_dense_v4_baseline__t{trial.number:06d}"
    run_dir = "experiments/sac_fetch_slide_dense_v4_baseline/runs/" + args.run_name
    return train(args, run_dir=run_dir, trial=trial)
