import numpy as np
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from gym_donkeycar.envs.donkey_env import * 
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy


training_observations = np.load("imitation_learning_datasets/imitation_training_data1/training_observations.npy", allow_pickle=True)
training_actions = np.load("imitation_learning_datasets/imitation_training_data1/training_actions.npy", allow_pickle=True)

trajectory_list = []
for index, (observations, actions) in enumerate(zip(training_observations, training_actions)):
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    trajectory_list.append(Trajectory(np.array(observations), np.array(actions), infos=None, terminal=True))
    
    
    
    
bc_trainer = bc.BC(
    observation_space=spaces.Box(
        shape=(363,),
        low= -np.inf,
        high= np.inf,
        dtype=np.float32,
    ),
    action_space=spaces.Box(
        shape=(2,),
        low=np.array([-1, -0.7]),
        high=np.array([1, 0.7]),
        dtype=np.float32,
    ),
    demonstrations=trajectory_list,
    rng = np.random.default_rng()
)


bc_trainer.train(n_epochs=300)

bc_trainer.policy.save("bc_policy_1000_epochs.zip")
