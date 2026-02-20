# Fetch Equivariant Sweep Plan: Push + PickAndPlace (2026-02-20)

## Sweep setup

Shared settings (both tasks):
- study direction: maximize eval return
- runtime: 24h CPU jobs
- resources: 8 CPU, 24G RAM
- array: 64 workers, 1 trial per worker
- objective env vars:
  - `FETCH_MODEL=equivariant`
  - `FETCH_EQ_SWEEP_MODE=pushpick_v1`

Studies:
- Push: `fetch_push_equivariant_v1_cpu_24h`
- PickAndPlace: `fetch_pick_equivariant_v1_cpu_24h`

## Hyperparameter space (`pushpick_v1`)

Reasoning: carry over slide findings while dropping expensive/unstable tails.

- `policy_lr`: `loguniform[2e-4, 8e-4]`
- `q_lr`: `loguniform[2e-4, 1.2e-3]`
- `gamma`: `uniform[0.97, 0.995]`
- `tau`: `loguniform[2e-3, 1.5e-2]`
- `batch_size`: `{128, 256}`
- `dihedral_n`: `{4, 8}`
- `reg_rep_n`: `{16, 32}`

## Launch commands (cluster)

```bash
cd ~/project/hypr_opt
sbatch sweep_push_equivariant_cpu_24h_v1.sbatch
sbatch sweep_pick_equivariant_cpu_24h_v1.sbatch
```

## Monitoring

```bash
squeue -u $USER -o "%i %j %T %M %l %R"
```

```bash
python - <<'PY'
import optuna
from common.optuna_db import make_storage
for name in [
    'fetch_push_equivariant_v1_cpu_24h',
    'fetch_pick_equivariant_v1_cpu_24h',
]:
    s = optuna.load_study(study_name=name, storage=make_storage())
    ts = s.get_trials(deepcopy=False)
    print(name, 'total', len(ts), {st.name: sum(t.state.name==st.name for t in ts) for st in optuna.trial.TrialState})
    if any(t.state.name=='COMPLETE' for t in ts):
        print('best', s.best_trial.number, s.best_value, s.best_params)
PY
```

## Dashboard

```bash
optuna-dashboard "$OPTUNA_STORAGE_URL" --port 8080
```

Then open the selected study and inspect:
- optimization history
- parameter importances
- slice plot
- parallel coordinate plot
- trial table (for stale RUNNING trials / crashes)
