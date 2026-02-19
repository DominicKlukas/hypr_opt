import os
from dataclasses import replace

import optuna

from algos.sac_fetch_equivariant import Args, train


def build_trial_args(base: Args, trial: optuna.Trial) -> Args:
    return replace(
        base,
        policy_lr=trial.suggest_float("policy_lr", 1e-4, 1e-3, log=True),
        q_lr=trial.suggest_float("q_lr", 1e-4, 2e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        tau=trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        # Narrowed for cluster practicality: avoids very slow/high-cost equivariant configs.
        batch_size=trial.suggest_categorical("batch_size", [128, 256]),
        dihedral_n=trial.suggest_categorical("dihedral_n", [4, 8]),
        reg_rep_n=trial.suggest_categorical("reg_rep_n", [16, 32, 64]),
    )


def run_objective(base: Args, trial: optuna.Trial, run_tag: str) -> float:
    args = build_trial_args(base, trial)
    args = replace(args, seed=1000 + trial.number)
    args.run_name = f"{run_tag}__t{trial.number:06d}"

    run_dir = os.path.join(os.path.dirname(__file__), run_tag, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    return train(args, run_dir=run_dir, trial=trial)
