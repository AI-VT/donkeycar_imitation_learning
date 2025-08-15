import numpy as np
from pynput import keyboard
from gym_donkeycar.envs.donkey_env import * 
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.data import serialize
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
# KEYBOARD CALLBACKS
# ==============================================================================

keyboard_throttle_command = 0.
keyboard_steering_command = 0.

def keyboard_press_callback(key: keyboard.KeyCode):
    global keyboard_steering_command, keyboard_throttle_command
    try:
        if key.char == "d":
            keyboard_steering_command = 1
        elif key.char == "a":
            keyboard_steering_command = -1
        elif key.char == "w":
            keyboard_throttle_command = 0.7
        elif key.char == "s":
            keyboard_throttle_command = -0.7
            
    except:
        pass

def keyboard_release_callback(key: keyboard.KeyCode):
    global keyboard_steering_command, keyboard_throttle_command
    
    try:
        if key.char == "d":
            keyboard_steering_command = 0.
        elif key.char == "a":
            keyboard_steering_command = 0.
        elif key.char == "w":
            keyboard_throttle_command = 0.
        elif key.char == "s":
            keyboard_throttle_command = 0.
            
    except:
        pass
    
    
listener = keyboard.Listener(on_press=keyboard_press_callback, on_release=keyboard_release_callback)
listener.start()




# ==============================================================================
# SET UP THE DONKEYCAR SIMULATOR
# ==============================================================================


donkeycar_simulation_config = {
    "exe_path" : PATH_TO_SIMULATOR_EXECUTABLE, 
    **DONKEYCAR_SIMULATION_CONFIG
}
env = DonkeyEnvLidar(level="mini_monaco", conf=donkeycar_simulation_config)

# This is the main array that contains data about all of the trajectories, which we will use as training data for the behavioural cloning
list_of_trajectories: list[Trajectory] = []



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
        
        action = np.array([keyboard_steering_command, keyboard_throttle_command])
        
        camera_observation, reward, done, info = env.step(action)
        lidar_observation = env.viewer.handler.lidar
        velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
    
        # Append the velocity and lidar observations together so the RL agent can see both of them
        observation = np.concatenate([lidar_observation, velocity_observation])
        current_lap_observations.append(observation)
        current_lap_actions.append(action)


        # If we finished the current lap
        if (info["lap_count"] == 1):                    
            list_of_trajectories.append(Trajectory(np.array(current_lap_observations), np.array(current_lap_actions), infos=None, terminal=True))  
            serialize.save(f"{IMITATION_LEARNING_DATASET_FOLDER}", list_of_trajectories)
            break
        
        # If we hit an object or we go too far off of the track, then we should not record the current lap
        if (abs(info["cte"]) > 8) or (info["hit"] != "none"):
            break

env.close()





# ==============================================================================
# TRAIN AND SAVE A BEHAVIOURAL CLONING ALGORITHM
# ==============================================================================
    

trajectory_list = serialize.load(f"{IMITATION_LEARNING_DATASET_FOLDER}")


bc_trainer = bc.BC(
    observation_space= env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectory_list,
    rng = np.random.default_rng()
)


bc_trainer.train(n_epochs=300)

bc_trainer.policy.save("imitation_learning_models/bc_policy_1000_epochs.zip")
