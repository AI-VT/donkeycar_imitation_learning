import numpy as np
from pynput import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym.envs.registration import register
from gym_donkeycar.envs.donkey_env import * 
from imitation.algorithms import bc




throttle_command = 0.
steering_command = 0.

def on_press(key: keyboard.KeyCode):
    global steering_command, throttle_command
    try:
        if key.char == "d":
            steering_command = 1
        elif key.char == "a":
            steering_command = -1
        elif key.char == "w":
            throttle_command = 0.7
        elif key.char == "s":
            throttle_command = -0.7
            
    except:
        pass
    


def on_release(key: keyboard.KeyCode):
    global steering_command, throttle_command
    
    try:
        if key.char == "d":
            steering_command = 0.
        elif key.char == "a":
            steering_command = 0.
        elif key.char == "w":
            throttle_command = 0.
        elif key.char == "s":
            throttle_command = 0.
            
    except:
        pass
    
    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object

# def make_donkey_env():
#     global port
    
#     conf = { "exe_path" : exe_path, "port" : port}
#     port+=1
#     return DonkeyEnv(conf=conf, level="mini_monaco")


# port = 9091
conf = {
    "exe_path" : f"/home/animated/Projects/donkeycar/DonkeySimLinux/donkey_sim.x86_64", 
    "port" : 9091, 
    "lidar_config": {"deg_per_sweep_inc": 1.0}, 
    "max_cte": 8
}

env = MiniMonacoEnv(conf=conf)

# env = make_vec_env(lambda: DonkeyEnv(conf=conf, level="generated-track"), n_envs=2)
# env = make_vec_env(make_donkey_env, n_envs=8)

# model = PPO("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo")


training_observations = []
training_actions = []
training_infos = []
training_rewards = []

current_lap_observations = []
current_lap_actions = []
current_lap_infos = []
current_lap_rewards = []

current_number_of_laps_completed = 0


camera_observation = env.reset()
lidar_observation = env.viewer.handler.lidar
velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
observation = np.concatenate([lidar_observation, velocity_observation])
current_lap_observations.append(observation)

for t in range(100000):
    
    action = np.array([steering_command, throttle_command])
    
    camera_observation, reward, done, info = env.step(action)
    lidar_observation = env.viewer.handler.lidar
    velocity_observation = np.array([env.viewer.handler.vel_x, env.viewer.handler.vel_y, env.viewer.handler.vel_z])
    observation = np.concatenate([lidar_observation, velocity_observation])
    
    current_lap_observations.append(observation)
    current_lap_actions.append(action)
    current_lap_infos.append(info)
    current_lap_rewards.append(reward)


    if (info["lap_count"] == 1):
        current_number_of_laps_completed += 1
        
        training_observations.append(current_lap_observations)
        training_actions.append(current_lap_actions)
        training_infos.append(current_lap_infos)
        training_rewards.append(current_lap_rewards)
        
        
        print(f"len observations: {len(current_lap_observations)}")
        print(f"len actions: {len(current_lap_actions)}") # TODO TEST EVERYTHING
        print(f"len infos: {len(current_lap_infos)}")
        print(f"len rewards: {len(current_lap_rewards)}")
        print(f"len training_observations: {len(training_observations)}")
        print(f"len training_actions: {len(training_actions)}")
        print(f"len training_infos: {len(training_infos)}")
        print(f"len training_rewards: {len(training_rewards)}")
        print()
        
        np.save('imitation_learning_datasets/imitation_training_data1/training_observations.npy', np.array(training_observations, dtype=object))
        np.save('imitation_learning_datasets/imitation_training_data1/training_actions.npy', np.array(training_actions, dtype=object))
        np.save('imitation_learning_datasets/imitation_training_data1/training_infos.npy', np.array(training_infos, dtype=object))
        np.save('imitation_learning_datasets/imitation_training_data1/training_rewards.npy', np.array(training_rewards, dtype=object))
        
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

