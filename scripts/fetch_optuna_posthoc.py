#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import median

import optuna

from common.optuna_db import make_storage


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize_study(study: optuna.Study, top_k: int, top_frac: float) -> dict:
    trials = study.get_trials(deepcopy=False)
    counts = Counter(t.state.name for t in trials)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    complete_sorted = sorted(complete, key=lambda t: float(t.value), reverse=True)

    summary = {
        "study": study.study_name,
        "total_trials": len(trials),
        "counts": dict(counts),
        "best_trial": None,
        "top_trials": [],
        "param_ranges_from_top": {},
        "arch_buckets": [],
    }

    if not complete_sorted:
        return summary

    best = complete_sorted[0]
    summary["best_trial"] = {
        "number": best.number,
        "value": float(best.value),
        "params": best.params,
    }

    for t in complete_sorted[:top_k]:
        summary["top_trials"].append(
            {
                "number": t.number,
                "value": float(t.value),
                "params": t.params,
            }
        )

    top_n = max(5, int(len(complete_sorted) * top_frac))
    top_set = complete_sorted[:top_n]

    param_values: dict[str, list] = defaultdict(list)
    for t in top_set:
        for k, v in t.params.items():
            param_values[k].append(v)

    for name, values in sorted(param_values.items()):
        if not values:
            continue
        if isinstance(values[0], (float, int)) and len({type(v) for v in values}) == 1 and isinstance(values[0], float):
            s = sorted(float(v) for v in values)
            summary["param_ranges_from_top"][name] = {
                "q10": quantile(s, 0.10),
                "q50": quantile(s, 0.50),
                "q90": quantile(s, 0.90),
                "min": min(s),
                "max": max(s),
            }
        elif isinstance(values[0], int):
            c = Counter(values)
            summary["param_ranges_from_top"][name] = {"counts": dict(sorted(c.items(), key=lambda kv: kv[0]))}
        else:
            c = Counter(str(v) for v in values)
            summary["param_ranges_from_top"][name] = {"counts": dict(c)}

    if all(any(p in t.params for p in ["dihedral_n", "reg_rep_n", "batch_size"]) for t in complete_sorted):
        buckets: dict[tuple, list[float]] = defaultdict(list)
        for t in complete_sorted:
            key = (t.params.get("dihedral_n"), t.params.get("reg_rep_n"), t.params.get("batch_size"))
            buckets[key].append(float(t.value))
        ranked = []
        for (d, r, b), vals in buckets.items():
            ranked.append(
                {
                    "dihedral_n": d,
                    "reg_rep_n": r,
                    "batch_size": b,
                    "n_complete": len(vals),
                    "mean": sum(vals) / len(vals),
                    "median": median(vals),
                    "best": max(vals),
                }
            )
        ranked.sort(key=lambda row: row["best"], reverse=True)
        summary["arch_buckets"] = ranked[:10]

    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['study']} ===")
    print("total_trials:", summary["total_trials"])
    print("states:", summary["counts"])
    if not summary["best_trial"]:
        print("best: none")
        return

    best = summary["best_trial"]
    print(f"best: trial={best['number']} value={best['value']:.6f}")
    print("best_params:", json.dumps(best["params"], sort_keys=True))

    print("top_trials:")
    for row in summary["top_trials"]:
        print(f"  t{row['number']:04d} value={row['value']:.6f} params={json.dumps(row['params'], sort_keys=True)}")

    print("suggested_tight_ranges_from_top:")
    for k, v in summary["param_ranges_from_top"].items():
        print(f"  {k}: {json.dumps(v, sort_keys=True)}")

    if summary["arch_buckets"]:
        print("top_arch_buckets:")
        for row in summary["arch_buckets"]:
            print(
                "  "
                f"d{row['dihedral_n']}_r{row['reg_rep_n']}_b{row['batch_size']} "
                f"n={row['n_complete']} mean={row['mean']:.4f} median={row['median']:.4f} best={row['best']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Posthoc analysis for Fetch Optuna studies.")
    parser.add_argument("--studies", nargs="+", required=True, help="One or more study names")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-frac", type=float, default=0.25)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if not os.environ.get("OPTUNA_STORAGE_URL"):
        raise SystemExit("OPTUNA_STORAGE_URL is not set")

    storage = make_storage()
    results = []
    for name in args.studies:
        study = optuna.load_study(study_name=name, storage=storage)
        summary = summarize_study(study, top_k=args.top_k, top_frac=args.top_frac)
        print_summary(summary)
        results.append(summary)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
