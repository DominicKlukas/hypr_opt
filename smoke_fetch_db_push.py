#!/usr/bin/env python3
import argparse
import os
import socket
import time
import uuid

import optuna

from common.optuna_db import make_storage, retry_db


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal smoke test for Optuna DB push (ask/tell + metadata)."
    )
    p.add_argument(
        "--study-name",
        default="smoke_fetch_db_push_min",
        help="Study name to create/load in remote storage.",
    )
    p.add_argument(
        "--value",
        type=float,
        default=-123.0,
        help="Constant objective value to write for each trial.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of minimal trials to push.",
    )
    p.add_argument(
        "--storage-url",
        default=None,
        help="Optional override for OPTUNA_STORAGE_URL.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    host = socket.gethostname()
    pid = os.getpid()
    ts = int(time.time())
    marker = f"smoke_db_push_{run_id}_{ts}"

    storage = make_storage(args.storage_url)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    for _ in range(args.n_trials):
        trial = retry_db(study.ask)

        def set_meta() -> None:
            trial.set_user_attr("smoke_marker", marker)
            trial.set_user_attr("host", host)
            trial.set_user_attr("pid", pid)
            trial.set_user_attr("run_id", run_id)
            trial.set_user_attr("where", os.environ.get("WHERE", "unknown"))

        retry_db(set_meta)
        retry_db(lambda: study.tell(trial, args.value))

    reloaded = optuna.load_study(study_name=args.study_name, storage=storage)
    trials = reloaded.get_trials(deepcopy=False)
    pushed = [
        t
        for t in trials
        if t.user_attrs.get("smoke_marker") == marker
        and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(pushed) < args.n_trials:
        raise RuntimeError(
            f"DB write verification failed: expected {args.n_trials} COMPLETE trial(s) "
            f"with marker={marker}, found {len(pushed)}."
        )

    print("SMOKE_DB_PUSH_OK")
    print("study_name =", args.study_name)
    print("marker =", marker)
    print("pushed_trials =", len(pushed))
    print("last_trial_number =", pushed[-1].number)


if __name__ == "__main__":
    main()
