#!/usr/bin/env python3
import os
import socket
import uuid

import optuna

from common.optuna_db import make_storage, retry_db
from experiments.sac_fetch_pick_and_place_dense_v4.objective import objective as objective_pick
from experiments.sac_fetch_push_dense_v4.objective import objective as objective_push
from experiments.sac_fetch_reach_dense_v4.objective import objective as objective_reach
from experiments.sac_fetch_slide_dense_v4.objective import objective as objective_slide
from experiments.sac_fetch_slide_dense_v4_baseline.objective import objective as objective_slide_baseline


RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
HOST = socket.gethostname()
PID = os.getpid()

TASK_TO_OBJECTIVE = {
    "equivariant": {
        "reach": objective_reach,
        "push": objective_push,
        "pick": objective_pick,
        "slide": objective_slide,
    },
    "baseline": {
        "slide": objective_slide_baseline,
    },
}


def run_one_trial(study: optuna.Study, objective_fn) -> None:
    trial = retry_db(study.ask)

    def set_meta():
        trial.set_user_attr("run_id", RUN_ID)
        trial.set_user_attr("host", HOST)
        trial.set_user_attr("pid", PID)
        trial.set_user_attr("task", os.environ.get("FETCH_TASK", "reach"))
        trial.set_user_attr("model", os.environ.get("FETCH_MODEL", "equivariant"))

    retry_db(set_meta)

    try:
        value = objective_fn(trial)
    except optuna.TrialPruned:
        retry_db(lambda: study.tell(trial, state=optuna.trial.TrialState.PRUNED))
        return
    except Exception:
        retry_db(lambda: study.tell(trial, state=optuna.trial.TrialState.FAIL))
        raise

    retry_db(lambda: study.tell(trial, value))


def main() -> None:
    model = os.environ.get("FETCH_MODEL", "equivariant").lower().strip()
    task = os.environ.get("FETCH_TASK", "reach").lower().strip()
    if model not in TASK_TO_OBJECTIVE:
        raise ValueError(f"Unknown FETCH_MODEL={model}. Use one of: {', '.join(TASK_TO_OBJECTIVE)}")
    if task not in TASK_TO_OBJECTIVE[model]:
        raise ValueError(f"Unknown FETCH_TASK={task} for model={model}. Use one of: {', '.join(TASK_TO_OBJECTIVE[model])}")

    storage = make_storage()
    study_name = os.environ.get("OPTUNA_STUDY", f"fetch_{task}_{model}")

    sampler = optuna.samplers.TPESampler(constant_liar=True, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(os.environ.get("PRUNER_STARTUP_TRIALS", "20")),
        n_warmup_steps=int(os.environ.get("PRUNER_WARMUP_STEPS", "2")),
        interval_steps=int(os.environ.get("PRUNER_INTERVAL_STEPS", "1")),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    n_trials = int(os.environ.get("N_TRIALS", "2"))
    objective_fn = TASK_TO_OBJECTIVE[model][task]

    for _ in range(n_trials):
        run_one_trial(study, objective_fn)

    print("task =", task)
    print("model =", model)
    print("best_value =", study.best_value)
    print("best_params =", study.best_params)


if __name__ == "__main__":
    main()
