import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from gym.envs.registration import register
from gym_donkeycar.envs.donkey_env import * 
from imitation.algorithms import bc
import torch

conf = {
    "exe_path" : f"/home/animated/Projects/donkeycar/DonkeySimLinux/donkey_sim.x86_64", 
    "port" : 9091, 
    "lidar_config": {"deg_per_sweep_inc": 1.0}, 
    "max_cte": 8
}

saved_variables = torch.load("bc_policy_1000_epochs.zip", weights_only=False)
model = ActorCriticPolicy(**saved_variables["data"])
model.load_state_dict(saved_variables["state_dict"])


env = MiniMonacoEnv(conf=conf)
# policy = ActorCriticPolicy.load("bc_policy_1000_epochs.zip")


camera_observation = env.reset()
lidar_observation = env.viewer.handler.lidar
velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
observation = np.concatenate([lidar_observation, velocity_observation])


for t in range(100000):
    
    action, _ = model.predict(observation, deterministic=True)
    print(action)
    
    camera_observation, reward, done, info = env.step(action)
    lidar_observation = env.viewer.handler.lidar
    velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
    observation = np.concatenate([lidar_observation, velocity_observation])


    if (info["lap_count"] == 1):
        current_number_of_laps_completed += 1
        camera_observation = env.reset()
        lidar_observation = env.viewer.handler.lidar
        velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
        observation = np.concatenate([lidar_observation, velocity_observation])
        
        current_lap_observations = []
        current_lap_actions = []
        current_lap_infos = []
        current_lap_rewards = []
        current_lap_observations.append(observation)
        
        
    if (abs(info["cte"]) > 8) or (info["hit"] != "none"):
        camera_observation = env.reset()
        lidar_observation = env.viewer.handler.lidar
        velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
        observation = np.concatenate([lidar_observation, velocity_observation])
        
        current_lap_observations = []
        current_lap_actions = []
        current_lap_infos = []
        current_lap_rewards = []
        current_lap_observations.append(observation)



# Exit the scene
env.close()

