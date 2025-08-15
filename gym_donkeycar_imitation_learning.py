import numpy as np
from imitation.algorithms import bc
from imitation.data import serialize
from imitation.data.types import Trajectory
from gym_donkeycar.envs.donkey_env import * 
from gymnasium import spaces
import yaml


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

GLOBAL_VARIABLES: dict = yaml.safe_load(open("global_variables.yaml", 'r'))

IMITATION_LEARNING_DATASET_FOLDER = GLOBAL_VARIABLES["imitation_learning_dataset_folder"]


# ==============================================================================
# TRAIN AND SAVE A BEHAVIOURAL CLONING ALGORITHM
# ==============================================================================
    

trajectory_list = serialize.load(f"{IMITATION_LEARNING_DATASET_FOLDER}")


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

bc_trainer.policy.save("imitation_learning_models/bc_policy_1000_epochs.zip")
