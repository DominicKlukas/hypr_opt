from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, args, run_dir):
        self.writer = SummaryWriter(f"{run_dir}/tb/{args.run_name}")

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str | None = None):
        for k, v in metrics.items():
            name = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(name, v, step)

    def log_scalar(self, key: str, value: float, step: int):
        self.writer.add_scalar(key, value, step)

    def log_text(self, key: str, text: str):
        self.writer.add_text(key, text)

    def close(self):
        self.writer.close()
