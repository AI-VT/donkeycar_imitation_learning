from gym_donkeycar.envs.donkey_env import DonkeyEnv
import numpy as np
from typing import Optional, Dict, Any, Tuple 
from gymnasium import spaces
import time


class DonkeyEnvCamera(DonkeyEnv):
    """
    A modified version of the DonkeyEnv that takes in a camera image.
    """
    
    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        super().__init__(level, conf)
        
        self.start_time = 0
        
        height, width, color = self.viewer.get_sensor_size()
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (color, height, width), dtype=np.uint8)
        
        # this contains the steering and throttle actions respectively
        self.action_space = spaces.Box(
            shape=(2,),
            low=-1,
            high=1,
            dtype=np.float32
        )
    
    def compute_reward(self, info: dict):
        speed = info.get("speed", 0)
        cte = abs(info.get("cte", 0))

        # Encourage forward progress and speed
        reward = info["forward_vel"] * 0.1

        # Penalize deviation from track center
        reward -= 0.05 * (cte ** 2)

        # Harsh penalty for leaving track or crashing
        if cte > 5.0:
            reward -= 3
        
        if (info["hit"] != "none"):
            reward -= 100

        if (info["lap_count"] == 1):                    
            reward += 50 - (time.time() - self.start_time)
        
        return reward


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        
        camera_observation, reward, done, info = super().step(action)
        
        reward = self.compute_reward(info=info)
        
        done = done or info["lap_count"] == 1 or (time.time() - self.start_time) > 40 # if we complete the lap or we are in the environment for too long then we are done

        camera_observation = np.moveaxis(camera_observation, -1, 0)
        return camera_observation, reward, done, info
    
    
    def reset(self, seed=None) -> tuple[np.ndarray, dict]:
        
        self.start_time = time.time()
        camera_observation = super().reset()
        
        camera_observation = np.moveaxis(camera_observation, -1, 0)

        return camera_observation
