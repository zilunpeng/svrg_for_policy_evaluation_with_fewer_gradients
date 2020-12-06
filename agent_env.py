from policy_iteration import *

def get_pi_env(env_type, exp_setup, is_loading, loading_path, is_saving, saving_path, policy_iteration_episode, init_method, num_data):
    if env_type == 'rmdp':
        return RandomMDP_Policy_Iteration(exp_setup=exp_setup, num_data=num_data, nStates=400, nActions=10, 
                                          nFeatures=200, reward_level=10, feature_level=1, gamma=0.95, rho=0, 
                                          num_states_reachable_from_sa=5, saving_path=saving_path, is_saving=is_saving, 
                                          loading_path=loading_path, is_loading=is_loading, 
                                          policy_iteration_episode=policy_iteration_episode, init_method=init_method)
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