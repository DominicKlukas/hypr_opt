import os, time, random
import optuna
import sqlalchemy
import psycopg2
from sqlalchemy.pool import NullPool
from typing import Callable, TypeVar, Any

T = TypeVar("T")

DEFAULT_STORAGE_URL = (
    "postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN"
    "@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require"
)

def make_storage(storage_url: str | None = None) -> optuna.storages.RDBStorage:
    url = (storage_url or os.environ.get("OPTUNA_STORAGE_URL", DEFAULT_STORAGE_URL)).strip()
    return optuna.storages.RDBStorage(
        url,
        engine_kwargs={"poolclass": NullPool, "pool_pre_ping": True},
        heartbeat_interval=30,
        grace_period=120,
    )

def is_pool_exhausted_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "MaxClientsInSessionMode" in msg
        or "max clients reached" in msg.lower()
        or "too many clients" in msg.lower()
        or "too many connections" in msg.lower()
    )

def retry_db(fn: Callable[[], T], *, max_retries: int = 200,
             base_sleep: float = 0.5, max_sleep: float = 60.0) -> T:
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

