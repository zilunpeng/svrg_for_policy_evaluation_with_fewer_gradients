"""

This script contains experiment set ups for results in Table 3. 

"""

import os

import pandas as pd

from experiment_Setup import Experiment_Setup
from agent_env import get_pi_env
from SVRG import *

if __name__ == '__main__':

    NUM_RUNS = 50

    # Random MDP (small data)
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1e-4, "sigma_omega": 1e-4, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     "record_per_epoch": False, 'num_checks': 10},

                     {"method": saga, "name": 'saga', "sigma_theta": 1e-2, "sigma_omega": 1e-1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},

                     {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1e-3, "sigma_omega": 1e-3, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1e-4, "sigma_omega": 1e-4, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": True,
                     "num_epoch": 100, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1e-3, 'sigma_omega': 1e-3, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'record_per_epoch': False, 'inner_loop_multiplier': 1,
                     "batch_svrg_init_ratio": 0.01, "batch_svrg_increment_ratio": 1.2, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=100, exp_settings=alg_settings, saving_dir_path="./", 
    	                             multi_process_exps=False, use_gpu=False, num_processes=1, 
    	                             batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="rmdp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
    	                    policy_iteration_episode=1, init_method="zero", num_data=20000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./rmdp_small_data_results.pkl')


    # Mountain Car (small data)
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     "record_per_epoch": False, 'num_checks': 10},

                     {"method": saga, "name": 'saga', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},

                     {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 2, 'record_per_epoch': False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": True,
                     "num_epoch": 100, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'record_per_epoch': False, 'inner_loop_multiplier': 1,
                     "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=100, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="mc", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=20000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./mc_small_data_results.pkl')


    # Cart Pole (small data)
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     "record_per_epoch": False, 'num_checks': 10},

                     {"method": saga, "name": 'saga', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},

                     {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": True,
                     "num_epoch": 100, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'record_per_epoch': False, 'inner_loop_multiplier': 1,
                     "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=100, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="cp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=20000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./cp_small_data_results.pkl')

    # Acrobot (small data)
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     "record_per_epoch": False, 'num_checks': 10},

                     {"method": saga, "name": 'saga', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 1, 'record_per_epoch': False, 'num_checks': 10},

                     {"method": svrg_classic, "name": 'svrg', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'inner_loop_multiplier': 2, 'record_per_epoch': False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": True,
                     "num_epoch": 100, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 0.1, 'sigma_omega': 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 100,
                     'record_per_epoch': False, 'inner_loop_multiplier': 1,
                     "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=100, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="ab", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=20000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./ab_small_data_results.pkl')

    NUM_RUNS = 20
    # Random MDP (large data)
    # We did not run SVRG and SAGA in this experiment because they require computing the full gradient and each method is only allowed to use the dataset once.
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1e-4, "sigma_omega": 1e-3, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     "record_per_epoch": False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1e-3, "sigma_omega": 1e-2, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.01, "record_per_dataset_pass": True,
                     "num_epoch": 1, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1e-3, 'sigma_omega': 1e-2, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     'record_per_epoch': False, 'inner_loop_multiplier': 0.05,
                     "batch_svrg_init_ratio": 0.01, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=1, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="rmdp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=1000000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./rmdp_large_data_results.pkl')


    # Mountain Car (large data)
    # We did not run SVRG and SAGA in this experiment because they require computing the full gradient and each method is only allowed to use the dataset once.
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     "record_per_epoch": False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.01, "record_per_dataset_pass": True,
                     "num_epoch": 1, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     'record_per_epoch': False, 'inner_loop_multiplier': 0.2,
                     "batch_svrg_init_ratio": 0.001, "batch_svrg_increment_ratio": 5, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=1, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="mc", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=1000000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./mc_large_data_results.pkl')


    # Cart Pole (large data)
    # We did not run SVRG and SAGA in this experiment because they require computing the full gradient and each method is only allowed to use the dataset once.
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     "record_per_epoch": False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.05, "record_per_dataset_pass": True,
                     "num_epoch": 1, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     'record_per_epoch': False, 'inner_loop_multiplier': 0.1,
                     "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=1, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="cp", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=1000000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./cp_large_data_results.pkl')


    # Acrobot (large data)
    # We did not run SVRG and SAGA in this experiment because they require computing the full gradient and each method is only allowed to use the dataset once.
    alg_settings = [
                    {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     "record_per_epoch": False, 'num_checks': 10},
            
                    {"method": scsg, "name": 'scsg', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
                     'scsg_batch_size_ratio': 0.05, "record_per_dataset_pass": True,
                     "num_epoch": 1, 'record_per_epoch': False, 'num_checks': 10},

                    {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 0.1, 'sigma_omega': 1, 'grid_search': False,
                     "record_per_dataset_pass": True, "num_epoch": 1,
                     'record_per_epoch': False, 'inner_loop_multiplier': 0.2,
                     "batch_svrg_init_ratio": 0.001, "batch_svrg_increment_ratio": 5, 'num_checks': 10},
    ]
    results = []
    for i in range(NUM_RUNS):
        exp_setup = Experiment_Setup(num_epoch=1, exp_settings=alg_settings, saving_dir_path="./", 
                                     multi_process_exps=False, use_gpu=False, num_processes=1, 
                                     batch_size=100, num_workers=0)
        pi_env = get_pi_env(env_type="ab", exp_setup=exp_setup, loading_path="", is_loading=False, saving_path="./", is_saving=True, 
                            policy_iteration_episode=1, init_method="zero", num_data=1000000)
        results.extend(pi_env.run_policy_iteration())
    pd.DataFrame(pi_results).to_pickle('./ab_large_data_results.pkl')