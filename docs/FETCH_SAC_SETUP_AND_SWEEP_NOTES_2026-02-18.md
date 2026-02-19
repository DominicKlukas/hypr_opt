# Fetch SAC Setup + Sweep Notes (2026-02-18)

## 1) What Was Added / Changed

### New algorithms
- `algos/sac_fetch_equivariant.py`
  - Refactored SAC for Fetch robotics tasks using equivariant networks.
  - Supports Optuna pruning via `PruneReporter`.
  - Supports `vector_env_mode` (`async` for cluster, `sync` for smoke tests).
- `algos/sac_fetch_baseline.py`
  - Non-equivariant MLP SAC baseline for fair comparison.
  - Mirrors training flow and tuned hyperparameters of the equivariant pipeline.
- `algos/dqn_snake_equivariant.py`
  - DQN equivariant module from Atari branch refactor.

### New network modules
- `networks/sac_equivariant.py`
- `networks/sac_mlp.py`
- `networks/dqn_snake_equivariant.py`
- `networks/__init__.py`
  - Made import-safe (`__all__ = []`) so baseline runs do not eagerly import equivariant deps.

### Common utilities
- `common/fetch_obs.py`
  - Relative-coordinate observation wrapper for Fetch Reach/Push/PickAndPlace/Slide.
- `common/replay_buffer.py`
  - Replay buffer adapted from SB3-style implementation.

### Experiment folders created
- `experiments/sac_fetch_reach_dense_v4/`
- `experiments/sac_fetch_push_dense_v4/`
- `experiments/sac_fetch_pick_and_place_dense_v4/`
- `experiments/sac_fetch_slide_dense_v4/`
- `experiments/sac_fetch_slide_dense_v4_baseline/`

Each includes `objective.py`; each Fetch folder also has `quick_run.py` smoke script.

### Worker entrypoint
- `optuna_worker_fetch.py`
  - Supports model/task selection with:
    - `FETCH_MODEL=equivariant|baseline`
    - `FETCH_TASK=reach|push|pick|slide` (baseline currently wired for `slide`)
  - Uses shared online DB storage from `common/optuna_db.py`.

### Dependency and quick-run setup
- Added `requirements.txt` (CPU-friendly torch index + main runtime deps).
- Added `experiments/dqn_snake_equivariant_dummy_quick_run.py` for DQN smoke validation.
- Updated PPO and Fetch quick runs to be smoke-test friendly in restricted environments.

### Recent commits
- `746a815` Add Fetch SAC equivariant/baseline experiments and quick-run deps
- `5f2e731` Avoid eager network imports and use explicit escnn module imports

---

## 2) Smoke Tests Performed

## Local (workspace) smoke status
Passed:
- PPO CartPole quick run
- PPO LunarLander quick run
- SAC Fetch quick runs (reach, push, pick-and-place, slide)
- DQN dummy quick run

## Cluster smoke status
### Baseline Slide quick run
- Passed after:
  - `module load mujoco`
  - Using cluster venv with `gymnasium_robotics` available.

### Equivariant Slide quick run
- Passed using:
  - `module load mujoco`
  - `PYTHONPATH=~/project/equivariant-rl:$PYTHONPATH` (to resolve vendored `escnn` stack on cluster)

Note: `escnn` from pip on cluster required `lie_learn` build path that failed there; vendored `escnn` path resolved this for smoke tests.

---

## 3) Sweep Spaces (Fair Comparison)

## Shared sweep dimensions (both models)
- `policy_lr`: `loguniform[1e-4, 1e-3]`
- `q_lr`: `loguniform[1e-4, 2e-3]`
- `gamma`: `uniform[0.95, 0.999]`
- `tau`: `loguniform[1e-3, 2e-2]`
- `batch_size`: `{128, 256, 512}`

## Equivariant-only dimensions
- `dihedral_n`: `{4, 8, 16}`
- `reg_rep_n`: `{32, 64, 128}`

## Rationale for fairness
- Both models share optimizer/discount/update-space search.
- Equivariant model adds architecture-specific knobs only where baseline has no analog.
- Training loop, evaluation cadence, and objective direction are kept aligned.

---

## 4) Online DB + Cluster Launch Commands

## Required before launching
- Ensure `OPTUNA_STORAGE_URL` is set to a valid DB URL (current fallback URL in `common/optuna_db.py` failed on cluster with `Tenant or user not found`).

Example export:
```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg2://...'
```

## Equivariant Slide study
```bash
cd ~/project/hypr_opt
source ~/project/.venv/bin/activate
module load mujoco
export PYTHONPATH=~/project/equivariant-rl:$PYTHONPATH
export FETCH_MODEL=equivariant
export FETCH_TASK=slide
export OPTUNA_STUDY=fetch_slide_equivariant
export N_TRIALS=5
python optuna_worker_fetch.py
```

## Baseline Slide study
```bash
cd ~/project/hypr_opt
source ~/project/.venv/bin/activate
module load mujoco
export FETCH_MODEL=baseline
export FETCH_TASK=slide
export OPTUNA_STUDY=fetch_slide_baseline
export N_TRIALS=5
python optuna_worker_fetch.py
```

For parallel trials, run many workers concurrently (e.g., Slurm array), each executing the same command with identical study name.

---

## 5) Known Cluster Constraints

- Interactive `salloc --gres=gpu:h100:1 ...` can queue for long periods depending on resources.
- Cluster pip resolver may build `mujoco` from source unless module-provided runtime is used.
- Equivariant stack is sensitive to `escnn` dependency path on cluster; vendored path via `PYTHONPATH` was the reliable smoke-tested path.

---

## 6) Current Readiness

- Code refactor + quick-run infrastructure: ready.
- Both Slide model families smoke-tested on cluster: yes.
- Online DB sweeps: blocked only by missing/invalid `OPTUNA_STORAGE_URL` value in cluster env.

---

## 7) DB Connection Mode Recommendation

For this project/workload, use **Transaction Pooler**.

Why (from project code assumptions):
- `common/optuna_db.py` uses a pooler host (`...pooler.supabase.com`) and aggressive reconnect/retry logic for transient connection issues.
- SQLAlchemy is configured with `NullPool`, which means many short-lived connections across workers rather than sticky session state.
- Retries explicitly handle pool exhaustion patterns (including session-mode-related limits), indicating the workload is designed for pooled, stateless DB interactions.
- Optuna worker pattern (`ask/tell` in short transactions) is well aligned with transaction pooling.

When to prefer other modes:
- **Session Pooler**: only if you need session-level state across requests (not typical here).
- **Direct connection**: only for low-concurrency admin/debug usage; not ideal for many concurrent workers.

---

## 8) Update Log

- 2026-02-18: Created this running document and switched to incremental updates by request.
- 2026-02-18: Added DB mode recommendation section (Transaction Pooler) with rationale tied to current Optuna DB code.

- 2026-02-18: Received explicit DB URL format from user and substituted password from previous project URL.
- 2026-02-18: Verified cluster DB connectivity after installing `psycopg2-binary` in cluster venv (`smoke_fetch_db_ok` study created).
- 2026-02-18: Submitted matched Slide sweeps on cluster (same resources, array size, and per-worker trial count):
  - Equivariant study: `fetch_slide_equivariant_v1`, job `23352325` (`slide_eq_opt`, array `1-8`, `N_TRIALS=5` per task).
  - Baseline study: `fetch_slide_baseline_v1`, job `23352326` (`slide_base_opt`, array `1-8`, `N_TRIALS=5` per task).
- 2026-02-18: Both jobs initially entered `PENDING` state in Slurm queue.

### Cluster Launch Details (executed)

Equivariant submit script: `sweep_slide_equivariant.sbatch`
- `--account=def-cneary`
- `--gres=gpu:h100:1`
- `--cpus-per-task=8`
- `--mem=24G`
- `--time=08:00:00`
- `--array=1-8`
- Env:
  - `FETCH_MODEL=equivariant`
  - `FETCH_TASK=slide`
  - `OPTUNA_STUDY=fetch_slide_equivariant_v1`
  - `N_TRIALS=5`

Baseline submit script: `sweep_slide_baseline.sbatch`
- Same Slurm resources and array settings as above.
- Env:
  - `FETCH_MODEL=baseline`
  - `FETCH_TASK=slide`
  - `OPTUNA_STUDY=fetch_slide_baseline_v1`
  - `N_TRIALS=5`

### Notes
- On cluster, `module load mujoco` is required before runs.
- `OPTUNA_STORAGE_URL` is passed explicitly in sbatch scripts.
- 2026-02-18 (status check): Slurm jobs `23352325` (equivariant) and `23352326` (baseline) remain `PENDING` with no started array tasks yet.

---

## 9) Context Handoff (for new Codex chat)

### What is currently running
- CPU sweep jobs (submitted on cluster):
  - Equivariant: `23364952` (`slide_eq_cpu`, array `1-64%12`)
  - Baseline: `23364953` (`slide_base_cpu`, array `1-64%12`)
- Old GPU-based jobs were canceled.

### Active studies
- Equivariant study name: `fetch_slide_equivariant_v2_cpu`
- Baseline study name: `fetch_slide_baseline_v2_cpu`

### DB connection used
- `OPTUNA_STORAGE_URL` set explicitly in sbatch scripts to:
  - `postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require`
- `psycopg2-binary` installed in cluster user site.

### Current observed progress (latest check)
- Equivariant trials are running and writing TensorBoard events.
- Some equivariant runs have first eval values (roughly -20 to -25 range).
- Baseline study had not yet appeared in DB at last check (likely scheduler/start-order effect).

### Cluster/runtime constraints discovered
- `module load mujoco` is required in jobs.
- For equivariant runs on cluster, `PYTHONPATH=~/project/equivariant-rl:$PYTHONPATH` is used so `escnn` imports work reliably.
- Stdout logs are sparse due to buffering; TensorBoard files are the reliable live signal.

### Key files to inspect first in new chat
- `docs/FETCH_SAC_SETUP_AND_SWEEP_NOTES_2026-02-18.md`
- `optuna_worker_fetch.py`
- `sweep_slide_equivariant_cpu.sbatch`
- `sweep_slide_baseline_cpu.sbatch`
- `experiments/sac_fetch_slide_dense_v4/objective.py`
- `experiments/sac_fetch_slide_dense_v4_baseline/objective.py`

### Quick status commands to run in new chat
```bash
# queue status
squeue -j 23364952,23364953 -o "%i %j %T %M %l %R"

# study progress
python - <<'PY'
import os, optuna
from common.optuna_db import make_storage
os.environ['OPTUNA_STORAGE_URL']='postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require'
for name in ['fetch_slide_equivariant_v2_cpu','fetch_slide_baseline_v2_cpu']:
    try:
        s=optuna.load_study(study_name=name, storage=make_storage())
    except Exception as e:
        print(name, 'NOT_FOUND', e)
        continue
    ts=s.get_trials(deepcopy=False)
    print(name, 'total', len(ts), {st.name:sum(t.state.name==st.name for t in ts) for st in optuna.trial.TrialState})
    if any(t.state.name=='COMPLETE' for t in ts):
        print('best', s.best_value, s.best_params)
PY
```

---

## 10) Minimal DB Push Smoke (No RL Training)

Use this to verify that workers can write trials to Optuna storage without involving MuJoCo, env setup, or policy code.

```bash
cd ~/project/hypr_opt
source ~/project/.venv/bin/activate
export OPTUNA_STORAGE_URL='postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require'
python smoke_fetch_db_push.py --study-name smoke_fetch_db_push_min --n-trials 1 --value -123.0
```

Expected success output includes:
- `SMOKE_DB_PUSH_OK`
- `study_name = ...`
- `marker = ...`
- `pushed_trials = 1`

If this passes while baseline slide workers still do not appear in DB, the issue is likely before/around baseline worker launch/import/runtime setup rather than DB connectivity.

---

## 11) Live Cluster Status Check (2026-02-19)

### Slurm status snapshot
- `23364952` (`slide_eq_cpu`): active; tasks `1-12` running, `13-64` pending due array task limit.
- `23364953` (`slide_base_cpu`): not in `squeue` because all array tasks have already failed.

### Study status snapshot
- `fetch_slide_equivariant_v2_cpu`: present with 12 `RUNNING`, 0 complete at check time.
- `fetch_slide_baseline_v2_cpu`: `NOT_FOUND` at check time.

### Root cause of baseline failure
- All baseline array tasks in `23364953` failed quickly (mostly 5-30s).
- Representative error (`logs/slide_base_cpu-23364953_1.err`):
  - `ModuleNotFoundError: No module named 'lie_learn'`
- Failure happens before any Optuna trial write for baseline because `optuna_worker_fetch.py` eagerly imported equivariant objectives (which import `escnn`/`lie_learn`) even when `FETCH_MODEL=baseline`.

### DB push smoke result (on cluster)
- Ran `python smoke_fetch_db_push.py --study-name smoke_fetch_db_push_min --n-trials 1 --value -123.0`
- Result: `SMOKE_DB_PUSH_OK` with one complete pushed trial.
- Conclusion: DB write path is working on cluster; baseline breakage is import/runtime path, not DB connectivity.
