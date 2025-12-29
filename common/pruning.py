# common/pruning.py
from dataclasses import dataclass
from typing import Optional
import optuna

@dataclass
class PruneReporter:
    trial: Optional[optuna.Trial]
    eval_interval_steps: int
    last_eval_step: int = 0
    eval_k: int = 0
    retry_db: Optional[callable] = None  # inject from worker

    def should_eval(self, global_step: int) -> bool:
        return (global_step - self.last_eval_step) >= self.eval_interval_steps

    def report(self, value: float, global_step: int) -> None:
        self.last_eval_step = global_step
        if self.trial is None:
            self.eval_k += 1
            return


        def _do_report():
            self.trial.report(value, step=self.eval_k)

        if self.retry_db is not None:
            self.retry_db(_do_report)
        else:
            _do_report()

        self.eval_k += 1

        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at eval_k={self.eval_k-1}, value={value}")

