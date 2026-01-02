#!/usr/bin/env python3
import os
import socket
import time
import uuid
import random

from common.pruning import PruneReporter
from common.optuna_db import make_storage, retry_db

import optuna
from sqlalchemy.pool import NullPool
import sqlalchemy
import psycopg2

from experiments.ppo_cartpole_v1.objective import objective


RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
HOST = socket.gethostname()
PID = os.getpid()

DEFAULT_STUDY = "supabase_test_2"

def run_one_trial_ask_tell(study: optuna.Study) -> None:
    trial = retry_db(study.ask)

    # Metadata (DB writes)
    def set_meta():
        trial.set_user_attr("run_id", RUN_ID)
        trial.set_user_attr("host", HOST)
        trial.set_user_attr("pid", PID)
        trial.set_user_attr("where", os.environ.get("WHERE", "local"))
        trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID", ""))
        trial.set_user_attr("slurm_array_task_id", os.environ.get("SLURM_ARRAY_TASK_ID", ""))

    retry_db(set_meta)

    try:
        value = objective(trial)  # <-- RL code owns report/prune
    except optuna.TrialPruned:
        retry_db(lambda: study.tell(trial, state=optuna.trial.TrialState.PRUNED))
        return
    except Exception:
        # Optional: mark failed explicitly (helps debugging)
        retry_db(lambda: study.tell(trial, state=optuna.trial.TrialState.FAIL))
        raise

    retry_db(lambda: study.tell(trial, value))


def main() -> None:
    storage = make_storage()
    study_name = os.environ.get("OPTUNA_STUDY", DEFAULT_STUDY)

    sampler = optuna.samplers.TPESampler(
        constant_liar=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(os.environ.get("PRUNER_STARTUP_TRIALS", "10")),
        n_warmup_steps=int(os.environ.get("PRUNER_WARMUP_STEPS", "3")),
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

    n_trials = int(os.environ.get("N_TRIALS", "5"))

    for _ in range(n_trials):
        run_one_trial_ask_tell(study)

    print("best_value =", study.best_value)
    print("best_params =", study.best_params)



if __name__ == "__main__":
    main()

