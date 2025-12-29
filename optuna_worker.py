#!/usr/bin/env python3
import os
import socket
import time
import uuid
import random

import optuna
from sqlalchemy.pool import NullPool
import sqlalchemy
import psycopg2

RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
HOST = socket.gethostname()
PID = os.getpid()

# Hardcoded default (env var OPTUNA_STORAGE_URL overrides this if set)
DEFAULT_STORAGE_URL = (
    "postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN"
    "@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require"
)

DEFAULT_STUDY = "supabase_test_3"


def objective(x) -> float:
    return (x - 2) ** 2


def make_storage() -> optuna.storages.RDBStorage:
    storage_url = os.environ.get("OPTUNA_STORAGE_URL", DEFAULT_STORAGE_URL).strip()
    return optuna.storages.RDBStorage(
        storage_url,
        engine_kwargs={
            "poolclass": NullPool,   # don't hoard connections (important for many workers)
            "pool_pre_ping": True,
        },
    )


def is_pool_exhausted_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "MaxClientsInSessionMode" in msg
        or "max clients reached" in msg.lower()
        or "too many clients" in msg.lower()
        or "too many connections" in msg.lower()
    )

"""
def optimize_with_retries(study: optuna.Study, n_trials: int) -> None:
    max_retries = int(os.environ.get("OPTUNA_MAX_RETRIES", "200"))
    base_sleep = float(os.environ.get("OPTUNA_RETRY_BASE_SLEEP", "0.5"))
    max_sleep = float(os.environ.get("OPTUNA_RETRY_MAX_SLEEP", "60"))

    completed = 0
    retries = 0

    while completed < n_trials:
        try:
            study.optimize(objective, n_trials=1)
            completed += 1
            retries = 0
        except (
            optuna.exceptions.StorageInternalError,
            sqlalchemy.exc.OperationalError,
            psycopg2.OperationalError,
        ) as e:
            if is_pool_exhausted_error(e):
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(f"Too many DB retry failures ({max_retries}).") from e

                sleep_s = min(max_sleep, base_sleep * (2 ** min(retries, 10)))
                sleep_s += random.uniform(0, 0.5)  # jitter
                time.sleep(sleep_s)
                continue
            raise
"""

def retry_db(fn, *, max_retries=200, base_sleep=0.5, max_sleep=60.0):
    retries = 0
    while True:
        try:
            return fn()
        except (
            optuna.exceptions.StorageInternalError,
            sqlalchemy.exc.OperationalError,
            psycopg2.OperationalError,
        ) as e:
            if not is_pool_exhausted_error(e):
                raise
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Too many DB retry failures ({max_retries}).") from e
            sleep_s = min(max_sleep, base_sleep * (2 ** min(retries, 10)))
            sleep_s += random.uniform(0, 0.5)
            time.sleep(sleep_s)


def main() -> None:
    storage = make_storage()
    study_name = os.environ.get("OPTUNA_STUDY", DEFAULT_STUDY)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )


    n_trials = int(os.environ.get("N_TRIALS", "5"))

    for _ in range(n_trials):
        trial = retry_db(study.ask)

        # 2) Set metadata + sample params (these are DB writes too, so retry them)
        def setup_and_sample():
            trial.set_user_attr("run_id", RUN_ID)
            trial.set_user_attr("host", HOST)
            trial.set_user_attr("pid", PID)
            trial.set_user_attr("where", os.environ.get("WHERE", "local"))
            trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID", ""))
            trial.set_user_attr("slurm_array_task_id", os.environ.get("SLURM_ARRAY_TASK_ID", ""))
            x = trial.suggest_float("x", -10, 10)
            return {"x": x}

        params = retry_db(setup_and_sample)

        value = objective(params['x'])

        retry_db(lambda: study.tell(trial, value))

    print("best_value =", study.best_value)
    print("best_params =", study.best_params)


if __name__ == "__main__":
    main()

