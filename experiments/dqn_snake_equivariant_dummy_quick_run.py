import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import gymnasium as gym

from algos.dqn_snake_equivariant import Args, train


class DummySnakeLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(12, 32, 32), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._steps += 1
        obs = self.observation_space.sample()
        reward = float(np.random.randn() * 0.01)
        terminated = self._steps >= 20
        truncated = False
        info = {}
        if terminated:
            info["episode"] = {"r": reward, "l": self._steps}
        return obs, reward, terminated, truncated, info


def make_env():
    return DummySnakeLikeEnv()


if __name__ == "__main__":
    args = Args(total_timesteps=500, learning_starts=50, eval_interval=200, eval_episodes=2, cuda=False)
    run_dir = os.path.join("experiments", "dqn_snake_equivariant_dummy", "runs", "quick_run")
    os.makedirs(run_dir, exist_ok=True)
    score = train(args, make_env=make_env, run_dir=run_dir, trial=None)
    print(f"quick_run_score={score}")
