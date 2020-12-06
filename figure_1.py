"""

This script contains experiment set ups for results in figure 1. 

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
                   {"method": svrg_classic, "name": "svrg", "sigma_theta": 1e-3, "sigma_omega": 1e-3,
                    "grid_search": False,  "record_per_dataset_pass": False, "record_per_epoch":True,
                    "num_epoch":50, "num_checks":10, "inner_loop_multiplier":1
                   },
                   {"method": batch_svrg, "name": 'batch_svrg', "sigma_theta": 1e-3, 'sigma_omega': 1e-3, 
                    "grid_search": False, "record_per_dataset_pass": False, "record_per_epoch": True,
                    "num_epoch": 50, "num_checks": 10, "inner_loop_multiplier": 1,
                    "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.05},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=50, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="rmdp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="zero", num_data=5000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./rmdp_results.pkl')


    # Mountain Car
    alg_settings = [
                   {"method": svrg_classic, "name": "svrg", "sigma_theta": 1e-1, "sigma_omega": 1e-1,
                    "grid_search": False,  "record_per_dataset_pass": False, "record_per_epoch":True,
                    "num_epoch":20, "num_checks":10, "inner_loop_multiplier":1
                   },
                   {"method": batch_svrg, "name": 'batch_svrg', "sigma_theta": 1e-1, 'sigma_omega': 1e-1, 
                    "grid_search": False, "record_per_dataset_pass": False, "record_per_epoch": True,
                    "num_epoch": 20, "num_checks": 10, "inner_loop_multiplier": 1,
                    "batch_svrg_init_ratio": 0.2, "batch_svrg_increment_ratio": 1.1},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=20, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="mc", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="zero", num_data=5000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./mc_results.pkl')


    # Cart Pole
    alg_settings = [
                   {"method": svrg_classic, "name": "svrg", "sigma_theta": 1, "sigma_omega": 1,
                    "grid_search": False,  "record_per_dataset_pass": False, "record_per_epoch":True,
                    "num_epoch":50, "num_checks":10, "inner_loop_multiplier":1
                   },
                   {"method": batch_svrg, "name": 'batch_svrg', "sigma_theta": 1, 'sigma_omega': 1, 
                    "grid_search": False, "record_per_dataset_pass": False, "record_per_epoch": True,
                    "num_epoch": 50, "num_checks": 10, "inner_loop_multiplier": 1,
                    "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.05},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=50, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="cp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="zero", num_data=5000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./cp_results.pkl')


    # Acrobot
    alg_settings = [
                   {"method": svrg_classic, "name": "svrg", "sigma_theta": 1e-2, "sigma_omega": 1e-2,
                    "grid_search": False,  "record_per_dataset_pass": False, "record_per_epoch":True,
                    "num_epoch":50, "num_checks":10, "inner_loop_multiplier":1
                   },
                   {"method": batch_svrg, "name": 'batch_svrg', "sigma_theta": 1e-2, 'sigma_omega': 1e-2, 
                    "grid_search": False, "record_per_dataset_pass": False, "record_per_epoch": True,
                    "num_epoch": 50, "num_checks": 10, "inner_loop_multiplier": 1,
                    "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.05},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=50, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="ab", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="random", num_data=5000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./ab_results.pkl')