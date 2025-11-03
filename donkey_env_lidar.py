from gym_donkeycar.envs.donkey_env import DonkeyEnv
import numpy as np
from typing import Optional, Dict, Any, Tuple 
from gymnasium import spaces
import time


class DonkeyEnvLidar(DonkeyEnv):
    """
    A modified version of the DonkeyEnv that takes in a lidar and velocity observations, which are often easier to learn from.
    """
    
    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        super().__init__(level, conf)
        
        self.start_time = 0
        number_of_lidar_measurements = int(round(360/ conf["lidar_config"]["deg_per_sweep_inc"]))
        
        print(f"number of lidar measurements: {number_of_lidar_measurements}")
        
        
        # for example, if there are 360 lidar measurements, then the observation space shape is (363,) because
        # it needs to fit 360 lidar measurements and the 3 velocity axes
        self.observation_space = spaces.Box(
            shape=(number_of_lidar_measurements+3,),
            low=-np.inf, 
            high=np.inf, 
            dtype=np.float32
        )
        
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
        
        lidar_observation = self.viewer.handler.lidar
        velocity_observation = np.array([self.viewer.handler.vel_x, self.viewer.handler.vel_y, self.viewer.handler.vel_z])
        
        # Append the velocity and lidar observations together so the RL agent can see both of them
        full_observation = np.concatenate([lidar_observation, velocity_observation])
        
        reward = self.compute_reward(info=info)
        
        done = done or info["lap_count"] == 1 or (time.time() - self.start_time) > 40 # if we complete the lap or we are in the environment for too long then we are done

        return full_observation, reward, done, done, info
    
    
    def reset(self, seed=None) -> tuple[np.ndarray, dict]:
        
        self.start_time = time.time()
        camera_observation = super().reset()
        
        lidar_observation = self.viewer.handler.lidar
        velocity_observation = np.array([self.viewer.handler.vel_x, self.viewer.handler.vel_y, self.viewer.handler.vel_z])
        
        # Append the velocity and lidar observations together so the RL agent can see both of them
        full_observation = np.concatenate([lidar_observation, velocity_observation])

        return full_observation, {}
