import optuna
from objective import objective

def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1, interval_steps=1)

    # local DB (sqlite) just for testing
    study = optuna.create_study(
        study_name="local_test",
        storage="sqlite:///experiments/ppo_cartpole_v1/local_optuna.db",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=10)
    print("best_value =", study.best_value)
    print("best_params =", study.best_params)

if __name__ == "__main__":
    main()

