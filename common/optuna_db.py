# common/optuna_db.py
import os
import time
import random
from typing import Callable, TypeVar

import optuna
import sqlalchemy
import psycopg2
from sqlalchemy.pool import NullPool

T = TypeVar("T")

DEFAULT_STORAGE_URL = (
    "postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN"
    "@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require"
)

def make_storage(storage_url: str | None = None) -> optuna.storages.RDBStorage:
    url = (storage_url or os.environ.get("OPTUNA_STORAGE_URL", DEFAULT_STORAGE_URL)).strip()
    return optuna.storages.RDBStorage(
        url,
        engine_kwargs={
            "poolclass": NullPool,
            "pool_pre_ping": True,
            "connect_args": {
                # Helps long-running cluster jobs survive transient network drops
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        },
        heartbeat_interval=30,
        grace_period=120,
    )

def is_pool_exhausted_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "maxclientsinsessionmode" in msg
        or "max clients reached" in msg
        or "too many clients" in msg
        or "too many connections" in msg
    )

def is_transient_conn_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = [
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection reset by peer",
        "connection timed out",
        "could not connect to server",
        "terminating connection",
        "eof detected",
        "broken pipe",
        "network is unreachable",
        "connection refused",
    ]
    return any(n in msg for n in needles)

def should_retry_db_error(exc: BaseException) -> bool:
    # Walk causes (Optuna often wraps SQLAlchemy/psycopg errors)
    cause = getattr(exc, "__cause__", None)
    if cause is not None and should_retry_db_error(cause):
        return True
    return is_pool_exhausted_error(exc) or is_transient_conn_error(exc)

def retry_db(
    fn: Callable[[], T],
    *,
    max_retries: int = 200,
    base_sleep: float = 0.5,
    max_sleep: float = 60.0,
) -> T:
    retries = 0
    while True:
        try:
            return fn()
        except (
            optuna.exceptions.StorageInternalError,
            sqlalchemy.exc.OperationalError,
            psycopg2.OperationalError,
        ) as e:
            if not should_retry_db_error(e):
                raise
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Too many DB retry failures ({max_retries}).") from e

            # exponential backoff + jitter (capped)
            sleep_s = min(max_sleep, base_sleep * (2 ** min(retries, 10)))
            sleep_s += random.uniform(0, 0.5)
            time.sleep(sleep_s)

