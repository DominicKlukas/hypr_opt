import optuna

from algos.sac_fetch_equivariant import Args
from experiments.sac_fetch_common import run_objective


def objective(trial: optuna.Trial) -> float:
    base = Args(
        env_id="FetchPickAndPlaceDense-v4",
        total_timesteps=1_000_000,
        eval_interval=25_000,
        eval_episodes=25,
        cuda=True,
        track=False,
    )
    return run_objective(
        base,
        trial,
        run_tag="sac_fetch_pick_and_place_dense_v4",
        sweep_mode="pushpick_v1",
    )
