#!/usr/bin/env python3
import os
import socket
import time
import uuid
import random

from common.pruning import PruneReporter

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

DEFAULT_STUDY = "supabase_test_2"


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


def simulate_rl_train(trial: optuna.Trial) -> float:
    # pretend hyperparams
    x = trial.suggest_float("x", -10, 10)

    reporter = PruneReporter(
        trial=trial,
        eval_interval_steps=int(os.environ.get("EVAL_INTERVAL_STEPS", "2000")),
        retry_db=retry_db,   # <-- important for DB-flaky environments
    )

    total_steps = int(os.environ.get("TOTAL_STEPS", "20000"))
    for global_step in range(1, total_steps + 1):
        # ... do training step ...

        if reporter.should_eval(global_step):
            # fake evaluation metric (lower is better here, like your toy objective)
            value = (x - 2.0) ** 2 + random.gauss(0.0, 0.2)
            reporter.report(value, global_step)

        # optional: simulate variable compute time
        if global_step % 1000 == 0:
            time.sleep(random.uniform(0.05, 0.2))

    # final score
    return (x - 2.0) ** 2



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
        value = simulate_rl_train(trial)  # <-- RL code owns report/prune
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
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    n_trials = int(os.environ.get("N_TRIALS", "10"))

    for _ in range(n_trials):
        run_one_trial_ask_tell(study)

    print("best_value =", study.best_value)
    print("best_params =", study.best_params)



if __name__ == "__main__":
    main()

