import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3 import PPO
from gym_donkeycar.envs.donkey_env import * 
import torch
import yaml
from donkey_env_lidar import DonkeyEnvLidar

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

GLOBAL_VARIABLES: dict = yaml.safe_load(open("global_variables.yaml", 'r'))

IMITATION_LEARNING_DATASET_FOLDER = GLOBAL_VARIABLES["imitation_learning_dataset_folder"]
PATH_TO_SIMULATOR_EXECUTABLE = GLOBAL_VARIABLES["path_to_simulator_executable"]
DONKEYCAR_SIMULATION_CONFIG = GLOBAL_VARIABLES["simulation_config"]



# ==============================================================================
# SET UP THE DONKEYCAR SIMULATOR
# ==============================================================================


donkeycar_simulation_config = {
    "exe_path" : PATH_TO_SIMULATOR_EXECUTABLE, 
    **DONKEYCAR_SIMULATION_CONFIG
}
env = MiniMonacoEnv(conf=donkeycar_simulation_config)


saved_variables = torch.load("imitation_learning_models/bc_policy_300_epochs.zip", weights_only=False)
model = ActorCriticPolicy(**saved_variables["data"])
# model = PPO("MlpPolicy", env, verbose=1)
model.load_state_dict(saved_variables["state_dict"])


# policy = ActorCriticPolicy.load("bc_policy_1000_epochs.zip")



# ==============================================================================
# RUN THE SIMULATOR WITH KEYBOARD CONTROL AND SAVE OBSERVATIONS AND ACTIONS
# ==============================================================================


for lap_index in range(1000):

    current_lap_observations = []
    current_lap_actions = []
    
    camera_observation = env.reset()
    lidar_observation = env.viewer.handler.lidar
    velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
    
    # Append the velocity and lidar observations together so the RL agent can see both of them
    observation = np.concatenate([lidar_observation, velocity_observation])
    current_lap_observations.append(observation)
            
            
    for time_step in range(10000000):
        
        action, _ = model.predict(observation, deterministic=True)
        
        camera_observation, reward, done, info = env.step(action)
        lidar_observation = env.viewer.handler.lidar
        velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
    
        # Append the velocity and lidar observations together so the RL agent can see both of them
        observation = np.concatenate([lidar_observation, velocity_observation])
        current_lap_observations.append(observation)
        current_lap_actions.append(action)


        # If we finished the current lap
        if (info["lap_count"] == 1):                    
            break
        
        # If we hit an object or we go too far off of the track, then we should not record the current lap
        if (abs(info["cte"]) > 8) or (info["hit"] != "none"):
            break

env.close()

