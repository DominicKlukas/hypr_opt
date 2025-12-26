import numpy as np
import torch

class Evaluate():
    def __init__(self, make):
        self.eval_env = make()

    @torch.no_grad()
    def evaluate(self, agent, device: torch.device, n_episodes: int, seed: int):
        """
        Simple deterministic eval for CartPole-style discrete action spaces.
        Reuses a single env by calling reset() each episode.
        Returns mean_return
        """
        agent.eval()
        returns = []

        for ep in range(n_episodes):
            obs, _ = self.eval_env.reset(seed=seed + ep)
            done = False
            info = {}

            while not done:
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.act(x)
                obs, r, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)

            returns.append(info["episode"]["r"])

        agent.train()
        print(f"Eval Return {np.mean(returns)}", flush=True)
        return float(np.mean(returns))

    def close(self):
        self.eval_env.close()

