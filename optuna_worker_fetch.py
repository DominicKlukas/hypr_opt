#!/usr/bin/env python3
import importlib
import os
import socket
import uuid

import optuna

from common.optuna_db import make_storage, retry_db


RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
HOST = socket.gethostname()
PID = os.getpid()

TASK_TO_MODULE_PATH = {
    "equivariant": {
        "reach": "experiments.sac_fetch_reach_dense_v4.objective",
        "push": "experiments.sac_fetch_push_dense_v4.objective",
        "pick": "experiments.sac_fetch_pick_and_place_dense_v4.objective",
        "slide": "experiments.sac_fetch_slide_dense_v4.objective",
    },
    "baseline": {
        "push": "experiments.sac_fetch_push_dense_v4_baseline.objective",
        "pick": "experiments.sac_fetch_pick_and_place_dense_v4_baseline.objective",
        "slide": "experiments.sac_fetch_slide_dense_v4_baseline.objective",
    },
}

def resolve_objective(model: str, task: str):
    module_path = TASK_TO_MODULE_PATH[model][task]
    module = importlib.import_module(module_path)
    return module.objective


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
    if model not in TASK_TO_MODULE_PATH:
        raise ValueError(f"Unknown FETCH_MODEL={model}. Use one of: {', '.join(TASK_TO_MODULE_PATH)}")
    if task not in TASK_TO_MODULE_PATH[model]:
        raise ValueError(f"Unknown FETCH_TASK={task} for model={model}. Use one of: {', '.join(TASK_TO_MODULE_PATH[model])}")

    storage = retry_db(make_storage)
    study_name = os.environ.get("OPTUNA_STUDY", f"fetch_{task}_{model}")

    sampler = optuna.samplers.TPESampler(constant_liar=True, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(os.environ.get("PRUNER_STARTUP_TRIALS", "20")),
        n_warmup_steps=int(os.environ.get("PRUNER_WARMUP_STEPS", "2")),
        interval_steps=int(os.environ.get("PRUNER_INTERVAL_STEPS", "1")),
    )

    study = retry_db(lambda: optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    ))

    jitter_max_s = float(os.environ.get("WORKER_STARTUP_JITTER_MAX_S", "0"))
    if jitter_max_s > 0:
        import random
        import time
        time.sleep(random.uniform(0, jitter_max_s))

    n_trials = int(os.environ.get("N_TRIALS", "2"))
    objective_fn = resolve_objective(model, task)

    for _ in range(n_trials):
        run_one_trial(study, objective_fn)

    print("task =", task)
    print("model =", model)
    print("best_value =", study.best_value)
    print("best_params =", study.best_params)


if __name__ == "__main__":
    main()
