import argparse
import numpy as np
from SVRG import *
from policy_iteration import *
from experiment_Setup import Experiment_Setup
import pandas as pd
import ast

def get_pi_env(env_type, exp_setup, is_loading, loading_path, is_saving, saving_path, policy_iteration_episode, init_method, num_data):
    if env_type == 'rmdp':
        return RandomMDP_Policy_Iteration(exp_setup=exp_setup, num_data=num_data, nStates=400, nActions=10, nFeatures=200, reward_level=10, feature_level=1, gamma=0.95, rho=0, num_states_reachable_from_sa=5, saving_path=saving_path, is_saving=is_saving, loading_path=loading_path, is_loading=is_loading, policy_iteration_episode=policy_iteration_episode, init_method=init_method)
    elif env_type == 'mc':
        return mountain_car_policy_iteration(exp_setup=exp_setup, feature_type='rbf', num_rbf_means=10,
                                             num_data=num_data, gamma=0.99, num_tilings=0, rbf_sigma=0.01, nFeatures=0,
                                             is_normalizing_rbf_feature=True, add_small_identity_to_A_C=False, saving_path=saving_path,
                                             is_saving=is_saving, loading_path=loading_path, is_loading=is_loading,
                                             policy_iteration_episode=policy_iteration_episode, init_method=init_method)
    elif env_type == 'cp':
        return Cart_Pole_Policy_Iteration(exp_setup=exp_setup, num_data=num_data, gamma=0.99, feature_type='rbf',
                                          rbf_sigma=0.5, reward_type='lspi_paper', is_normalizing_rbf_feature=True,
                                          add_const_one_to_feature=False, add_small_identity_to_A_C=False, saving_path=saving_path, is_saving=is_saving,
                                          loading_path=loading_path, is_loading=is_loading,
                                          policy_iteration_episode=policy_iteration_episode, init_method=init_method)
    elif env_type == 'ab':
        return Acrobot_Policy_Iteration(exp_setup=exp_setup, num_data=num_data, gamma=0.9, rbf_sigma=0.1,
                                        feature_type='rbf', num_rbf_means=3, add_const_one_to_feature=False,
                                        is_normalizing_rbf_feature=True, add_small_identity_to_A_C=False, saving_path=saving_path, is_saving=is_saving,
                                        loading_path=loading_path, is_loading=is_loading,
                                        policy_iteration_episode=policy_iteration_episode, init_method=init_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_saving_path", default=None)
    parser.add_argument("--loading_data_path", default=None)
    parser.add_argument("--policy_iter_environment", default=None)
    parser.add_argument("--num_epochs", default=None)
    parser.add_argument("--num_runs", default=None)
    parser.add_argument("--num_processes", default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--policy_iteration_episode", default=None)
    parser.add_argument("--record_per_epoch", default=None)
    parser.add_argument("--record_per_dataset_pass", default=None)
    parser.add_argument("--init_method", default=None)
    parser.add_argument("--num_data", default=None)
    args = parser.parse_args()
    num_epochs = int(args.num_epochs)
    num_runs = int(args.num_runs)
    num_processes = int(args.num_processes)
    batch_size = int(args.batch_size)
    policy_iteration_episode = int(args.policy_iteration_episode)
    record_per_epoch = ast.literal_eval(args.record_per_epoch)
    record_per_dataset_pass = ast.literal_eval(args.record_per_dataset_pass)
    terminate_if_less_than_epsilon = False
    policy_eval_epsilon = 1
    init_method = args.init_method
    num_data = int(args.num_data)

    #for rmdp
    if args.policy_iter_environment == 'rmdp':
        exp_settings = [
            # {"method": gtd2, "name": 'gtd2', "sigma_theta": 1e-4, "sigma_omega": 1e-4, 'grid_search': False,
            #  "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
            #  'record_per_epoch': record_per_epoch, 'num_checks': 10},
            #
            # {"method": scsg, "name": 'scsg', "sigma_theta": 1e-4, "sigma_omega": 1e-4, 'grid_search': False,
            #  'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": record_per_dataset_pass,
            #  "num_epoch": num_epochs, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            # {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1e-3, "sigma_omega": 1e-3, 'grid_search': False,
            #  "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
            #  'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            # {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1e-3, 'sigma_omega': 1e-3, 'grid_search': False,
            #  "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
            #  'record_per_epoch': record_per_epoch, 'inner_loop_multiplier': 1,
            #  "batch_svrg_init_ratio": 0.01, "batch_svrg_increment_ratio": 1.2, 'num_checks': 10},

            {"method": saga, "name": 'saga', "sigma_theta": 1e-2, "sigma_omega": 1e-1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},
        ]
    elif args.policy_iter_environment == 'mc':
        # for mc
        exp_settings = [
            {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": record_per_dataset_pass,
             "num_epoch": num_epochs, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 2, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": saga, "name": 'saga', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'inner_loop_multiplier': 1,
             "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
        ]
    elif args.policy_iter_environment == 'cp':
        # for cp
        exp_settings = [
            {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": scsg, "name": 'scsg', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
             'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": record_per_dataset_pass,
             "num_epoch": num_epochs, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": svrg_classic, "name": 'svrg', "sigma_theta": 1, "sigma_omega": 10, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": saga, "name": 'saga', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 1, 'sigma_omega': 10, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'inner_loop_multiplier': 1,
             "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
        ]
    elif args.policy_iter_environment == 'ab':
        # for cp
        exp_settings = [
            {"method": gtd2, "name": 'gtd2', "sigma_theta": 1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": scsg, "name": 'scsg', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
             'scsg_batch_size_ratio': 0.1, "record_per_dataset_pass": record_per_dataset_pass,
             "num_epoch": num_epochs, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": svrg_classic, "name": 'svrg', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 2, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": saga, "name": 'saga', "sigma_theta": 0.1, "sigma_omega": 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'inner_loop_multiplier': 1, 'record_per_epoch': record_per_epoch, 'num_checks': 10},

            {"method": batch_svrg, "name": 'batch_svrg', 'sigma_theta': 0.1, 'sigma_omega': 1, 'grid_search': False,
             "record_per_dataset_pass": record_per_dataset_pass, "num_epoch": num_epochs,
             'record_per_epoch': record_per_epoch, 'inner_loop_multiplier': 1,
             "batch_svrg_init_ratio": 0.1, "batch_svrg_increment_ratio": 1.1, 'num_checks': 10},
        ]

    root_saving_path = os.path.join(args.root_saving_path, args.loading_data_path)
    os.mkdir(root_saving_path)

    multi_process_exps = True
    use_gpu = True
    pi_results = []
    for i in range(num_runs):
        saving_dir_path = os.path.join(root_saving_path,'run'+str(i)+'/')
        os.mkdir(saving_dir_path)
        exp_setup = Experiment_Setup(num_epoch=100, exp_settings=exp_settings, saving_dir_path=saving_dir_path, multi_process_exps=multi_process_exps, use_gpu=use_gpu, num_processes=num_processes, batch_size=batch_size, num_workers=0)
        pi_env = get_pi_env(env_type=args.policy_iter_environment, exp_setup=exp_setup, loading_path=root_saving_path, is_loading=True, saving_path=saving_dir_path, is_saving=True, policy_iteration_episode=policy_iteration_episode, init_method=init_method, num_data=num_data)
        pi_results_one_epoch = pi_env.run_policy_iteration()
        pi_results.extend(pi_results_one_epoch)
        pd.DataFrame(pi_results_one_epoch).to_pickle(saving_dir_path + 'pi_results.pkl')
    pd.DataFrame(pi_results).to_pickle(root_saving_path+'/all_pi_results.pkl')

