"""

This script contains experiment set ups for results in figure 2.

We did not run SVRG and SAGA in this experiment because they require computing the full gradient and each method is only allowed to use the dataset once. 

"""

import os

import pandas as pd

from experiment_Setup import Experiment_Setup
from agent_env import get_pi_env
from SVRG import *

if __name__ == '__main__':

    NUM_RUNS = 10

    # Random MDP
    alg_settings = [
                   {"method": gtd2, "name": 'gtd2', "sigma_theta": 1e-5, "sigma_omega": 1e-4, 'grid_search': False,
                    "record_per_dataset_pass": True, "num_epoch": 1,
                    "record_per_epoch": False, 'num_checks': 10
                   },
                   {"method": scsg, "name": 'scsg', "sigma_theta": 1e-3, "sigma_omega": 1e-3, 'grid_search': False,
                    'scsg_batch_size_ratio': 0.001, "record_per_dataset_pass": True,
                    "num_epoch": 1, 'record_per_epoch': False, 'num_checks': 10},
                   {"method": batch_svrg, "name": 'batch_svrg', "sigma_theta": 1e-3, 'sigma_omega': 1e-3, 
                    "grid_search": False, "record_per_dataset_pass": True, "record_per_epoch": False,
                    "num_epoch": 1, "num_checks": 10, "inner_loop_multiplier": 1,
                    "batch_svrg_init_ratio": 0.001, "batch_svrg_increment_ratio": 1.1},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=1, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="rmdp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="zero", num_data=10000000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./rmdp_results.pkl')
