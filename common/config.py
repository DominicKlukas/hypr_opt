# common/config.py
import time
import os

def finalize_args(args, filename: str):
    """
    Mutates args:
    - computes batch sizes
    - computes number of iterations with batch sizes and time steps
    """
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    exp_name = os.path.basename(filename).replace(".py", "")
    args.exp_name = getattr(args, "exp_name", exp_name)  # optional
    return args

def setup_wandb(args, RUN_ID, trial=None):
    if not args.track:
        return
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.run_name,                 # unique per trial
        group=f"{args.env_id}__{args.exp_name}__{RUN_ID}",  # all trials together
        tags=[args.env_id, f"trial={trial.number}" if trial else "manual"],
        sync_tensorboard=True,
        config=vars(args) | ({"trial_number": trial.number} if trial else {}),
        monitor_gym=True,
        save_code=True,
        reinit=True,
    )
