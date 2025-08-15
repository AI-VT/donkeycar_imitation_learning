from gym_donkeycar.envs.donkey_env import DonkeyEnv
import numpy as np
from typing import Optional, Dict, Any, Tuple 
from gymnasium import spaces

class DonkeyEnvLidar(DonkeyEnv):
    """
    A modified version of the DonkeyEnv that takes in a lidar and velocity observations, which are often easier to learn from.
    """
    
    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        super().__init__(level, conf)
        
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
        

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        camera_observation, reward, done, info = super().step(action)
        
        lidar_observation = self.viewer.handler.lidar
        velocity_observation = np.array([self.viewer.handler.vel_x, self.viewer.handler.vel_y, self.viewer.handler.vel_z])
        
        # Append the velocity and lidar observations together so the RL agent can see both of them
        full_observation = np.concatenate([lidar_observation, velocity_observation])
        
        return full_observation, reward, done, info
    
    
    def reset(self) -> np.ndarray:
        camera_observation = super().reset()
        
        lidar_observation = self.viewer.handler.lidar
        velocity_observation = np.array([self.viewer.handler.vel_x, self.viewer.handler.vel_y, self.viewer.handler.vel_z])
        
        # Append the velocity and lidar observations together so the RL agent can see both of them
        full_observation = np.concatenate([lidar_observation, velocity_observation])

        return full_observation
