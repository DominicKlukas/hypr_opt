# common/config.py
import time
import os

def finalize_args(args, filename: str):
    """
    Mutates args:
    - computes batch sizes
    - sets run_name
    """
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    exp_name = os.path.basename(filename).replace(".py", "")
    args.exp_name = getattr(args, "exp_name", exp_name)  # optional
    if not getattr(args, "run_name", None) or args.run_name == "run":
        args.run_name = f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"

    return args


def setup_wandb(args):
    """
    Optional WandB + TensorBoard setup.
    Returns a SummaryWriter.
    """

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

