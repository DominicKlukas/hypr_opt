import numpy as np
import gymnasium as gym


class FetchObsWrapper(gym.ObservationWrapper):
    """Convert Fetch dict observations into a compact relative-coordinate vector.

    Output shapes:
    - FetchReach*: 6
    - FetchPush*/FetchPickAndPlace*/FetchSlide*: 19
    """

    def __init__(self, env: gym.Env, env_name: str):
        super().__init__(env)
        self.env_name = env_name
        self.obs_dim = self._compute_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _compute_obs_dim(self) -> int:
        if "FetchReach" in self.env_name:
            return 6
        if (
            "FetchPush" in self.env_name
            or "FetchPickAndPlace" in self.env_name
            or "FetchSlide" in self.env_name
        ):
            return 19
        raise ValueError(f"Unknown Fetch task: {self.env_name}")

    def observation(self, obs):
        o = obs["observation"]
        desired = obs["desired_goal"]

        if "FetchReach" in self.env_name:
            ee_abs = o[0:3]
            goal_rel = ee_abs - desired
            ee_vel = o[5:8]
            return np.concatenate([goal_rel, ee_vel]).astype(np.float32)

        if (
            "FetchPush" in self.env_name
            or "FetchPickAndPlace" in self.env_name
            or "FetchSlide" in self.env_name
        ):
            ee_abs = o[0:3]
            goal_rel = ee_abs - desired
            object_rel = o[6:9]
            ee_vel = o[20:23]
            grip_pos = o[9:11]
            grip_vel = o[23:]
            object_rot = o[11:14]
            object_vel = o[14:17]
            return np.concatenate(
                [goal_rel, object_rel, ee_vel, object_rot, object_vel, grip_pos, grip_vel]
            ).astype(np.float32)

        raise ValueError(f"Unknown Fetch task: {self.env_name}")
