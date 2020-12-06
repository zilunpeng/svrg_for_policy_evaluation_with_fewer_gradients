import numpy as np
from experiment_Setup import *
import gym
import h5py
import progressbar
from MDP_env import RandomMDP

class policy_iteration:
    def __init__(self,
                 exp_setup=None,
                 policy_iteration_episode=0, 
                 num_data=20000, gamma=0.95,
                 is_saving=False, saving_path='',
                 is_loading=False, loading_path='',
                 init_method='random'):
        self.exp_setup = exp_setup
        self.policy_iteration_episode = policy_iteration_episode
        self.num_data = num_data
        self.gamma = gamma
        self.is_saving = is_saving
        self.loading_path = loading_path
        self.is_loading = is_loading
        self.saving_data_path = saving_path
        self.init_method = init_method

    def create_exp_saving_dir(self):
        for exp_setting in self.exp_setup.exp_settings:
            method_id = self.get_method_id_from_dict(exp_setting)
            path = os.path.join(self.saving_data_path, method_id)
            try:
                os.mkdir(path)
            except OSError:
                pass
            exp_setting['saving_dir_path'] = path

    def get_lstd_soln(self,A, b, C_inv, nFeatures):
        A_T_C_inv = np.transpose(A) @ C_inv
        A_T_C_inv_A = A_T_C_inv @ A
        rank_A_T_C_inv_A = matrix_rank(A_T_C_inv_A)
        if rank_A_T_C_inv_A < nFeatures:
            A_T_C_inv_A = A_T_C_inv_A + np.identity(nFeatures) * 1e-7
            rank_A_T_C_inv_A = matrix_rank(A_T_C_inv_A)
        print('rank of A_T_C_inv_A is ' + str(rank_A_T_C_inv_A))
        A_T_C_inv_A_inv = np.linalg.solve(A_T_C_inv_A, np.identity(nFeatures)) if rank_A_T_C_inv_A == nFeatures else np.linalg.lstsq(A_T_C_inv_A, np.identity(nFeatures))
        theta = A_T_C_inv_A_inv @ A_T_C_inv @ b
        return theta

    def get_method_id_from_dict(self, dict):
        method_name = dict['name']
        step_sizes_info = '_theta_ss=' + str(dict['sigma_theta']) + '_omega_ss='+str(dict['sigma_omega'])
        if method_name == 'svrg':
            return method_name + '_inner_loop='+str(dict['inner_loop_multiplier']) + step_sizes_info
        elif method_name == 'scsg':
            return method_name + '_ratio='+str(dict['scsg_batch_size_ratio']) + step_sizes_info
        elif method_name == 'gtd2':
            return method_name + step_sizes_info
        elif method_name == 'batch_svrg':
            return method_name + '_init_ratio='+str(dict['batch_svrg_init_ratio'])+'_increment_ratio='+str(dict['batch_svrg_increment_ratio']) + step_sizes_info
        elif method_name == 'pdbg':
            return method_name + step_sizes_info
        elif method_name == 'saga':
            return method_name + step_sizes_info
        elif method_name == 'classical_td':
            return method_name + step_sizes_info + '_lambda='+str(dict['et_lambda'])
        elif method_name == 'tdc':
            return method_name + step_sizes_info + '_lambda='+str(dict['et_lambda'])
        elif method_name == 'etd':
            return method_name + step_sizes_info + '_lambda='+str(dict['et_lambda'])

    def get_policy(self, method_id):
        return self.policies[method_id]

    def update_policy(self, new_policy, method_id):
        self.policies[method_id] = new_policy

    def load_mdp_info(self, path):
        mdp_info = pd.read_pickle(os.path.join(self.loading_path,'mdp_info.pkl'))
        for attr in mdp_info:
            setattr(self, attr, mdp_info.at[0, attr])

    def save_mdp_info(self, path, mdp_info):
        pd.DataFrame(data=mdp_info, index=[0]).to_pickle(os.path.join(path,'mdp_info.pkl'))
        mdp_info_txt_fo = open(os.path.join(path,'mdp_info.txt'), "w")
        for k, v in mdp_info.items():
            mdp_info_txt_fo.write(str(k) + ' >>> ' + str(v) + '\n')
        mdp_info_txt_fo.close()

    def save_data_from_dict(self, path, data):
        for key,value in data.items():
            np.save(os.path.join(path, key), value)

    def save_data_for_policy_iteration(self, path, data):
        self.save_data_from_dict(path, data)
        shutil.copy(os.path.join(self.saving_data_path, 'mdp_info.pkl'), path)

    def parse_results(self, policy_eval_results):
        parsed_result = []
        new_exp_settings = []
        for result in policy_eval_results:
            if isinstance(result['result'], str) == False and np.any(np.isnan(result['theta'])) == False:
                method_id = self.get_method_id_from_dict(result)
                parsed_result.append(result)
                for exp_setting in self.exp_setup.exp_settings:
                    if self.get_method_id_from_dict(exp_setting) == method_id: new_exp_settings.append(exp_setting)
        return parsed_result, new_exp_settings

    def init_policies(self, policy_eval_results):
        self.policies = {}
        self.policy_scores = {}
        self.policy_weights = {}
        self.policy_compute_costs = {}
        self.all_policy_eval_results = {}
        for policy_eval_result in policy_eval_results:
            method_id = self.get_method_id_from_dict(policy_eval_result)
            policy = policy_eval_result['theta']
            self.policies[method_id] = policy
            self.policy_weights[method_id] = [policy]
            self.policy_scores[method_id] = [self.get_policy_score(policy)]
            print(method_id + ' . Policy score = ' + str(self.policy_scores[method_id]))
            self.policy_compute_costs[method_id] = [0]
            self.all_policy_eval_results[method_id] = [policy_eval_result['result']]

    def save_info_and_update_policy(self, policy_eval_results):
        for policy_eval_result in policy_eval_results:
            method_id = self.get_method_id_from_dict(policy_eval_result)
            policy_weights = policy_eval_result['theta']
            policy_score = self.get_policy_score(policy_weights)
            print(method_id + ' . Policy score = ' + str(policy_score))
            self.policy_scores[method_id].append(policy_score)
            self.policy_weights[method_id].append(policy_weights)
            self.policy_compute_costs[method_id].append(policy_eval_result['comp_cost'])
            self.all_policy_eval_results[method_id].append(policy_eval_result['result'])
            self.update_policy(policy_weights, method_id)

class continuous_env_policy_iteration(policy_iteration):
    def __init__(self, feature_type, num_rbf_means, rbf_sigma, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, add_const_one_to_feature, state_dim, init_method, add_small_identity_to_A_C):
        policy_iteration.__init__(self, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, init_method)
        self.feature_type = feature_type
        self.state_dim = state_dim
        self.num_rbf_means = num_rbf_means
        self.rbf_sigma = rbf_sigma
        self.explore_epsilon = 0.2
        self.add_const_one_to_feature = add_const_one_to_feature
        self.add_small_identity_to_A_C = add_small_identity_to_A_C

    def run_lspi(self):
        self.create_exp_saving_dir()
        self.load_and_save_data_for_all_settings() if self.num_data<100000 else self.load_and_save_large_data_for_all_settings()
        policy_eval_results = self.exp_setup.run_exps(self.exp_setup.exp_settings)
        policy_eval_results, exp_settings = self.parse_results(policy_eval_results)
        self.init_policies(policy_eval_results)
        for i in range(self.policy_iteration_episode):
            self.improve_policy(policy_eval_results, i)
            policy_eval_results = self.exp_setup.run_exps(exp_settings)
            policy_eval_results, exp_settings = self.parse_results(policy_eval_results)
            self.save_info_and_update_policy(policy_eval_results)
            self.explore_epsilon =  max(self.explore_epsilon*0.5, 1e-5)
        return_vals = []
        for result in policy_eval_results:
            temp = {}
            method_id = self.get_method_id_from_dict(result)
            temp['method_id'] = method_id
            temp['policy_scores'] = self.policy_scores[method_id]
            temp['comp_costs'] = self.policy_compute_costs[method_id]
            temp['pol_eval_results'] = self.all_policy_eval_results[method_id]
            temp['policy_weights'] = self.policy_weights[method_id]
            if 'record_points_before_one_pass' in result: temp['record_points_before_one_pass'] = result['record_points_before_one_pass']
            return_vals.append(temp)
        return_vals.append(self.get_lstd_soln(exp_settings))
        return return_vals

    def improve_policy(self, policy_eval_results, policy_iter_episode):
        for policy_eval_result in policy_eval_results:
            method_id = self.get_method_id_from_dict(policy_eval_result)
            loading_path = self.saving_data_path+method_id
            A, new_Phi = self.run_lstdq(policy_weights=policy_iteration.get_policy(self, method_id), trans_data=np.load(loading_path+'/Trans_data.npy'), next_states=np.load(loading_path+'/next_states.npy'), old_Phi=np.load(loading_path+'/Phi.npy'))
            np.save(loading_path+'/A.npy', A)
            np.save(loading_path+'/A_iter'+ str(policy_iter_episode) +'.npy', A)
            np.save(loading_path+'/Phi.npy', new_Phi)
            if self.init_method == 'random':
                np.save(loading_path+'/init_theta.npy', np.random.rand(self.nFeatures) * self.init_theta_omega_multiplier)
                np.save(loading_path+'/init_omega.npy', np.random.rand(self.nFeatures) * self.init_theta_omega_multiplier)
            elif self.init_method == 'previous':
                np.save(loading_path+'/init_theta.npy', policy_eval_result['theta'])
                np.save(loading_path+'/init_omega.npy', policy_eval_result['omega'])
            elif self.init_method == 'zero':
                np.save(loading_path + '/init_theta.npy', np.zeros(self.nFeatures))
                np.save(loading_path + '/init_omega.npy', np.zeros(self.nFeatures))
            np.save(loading_path+'/policy_weights.npy', policy_eval_result['theta'])
            np.save(loading_path+'/mspbe_results.npy', policy_eval_result['result'])

    def run_lstdq(self, policy_weights=None, trans_data=None, next_states=None, old_Phi=None):
        A, _, _ = MountainCar.init_abc(self.nFeatures)
        new_Phi = np.zeros(old_Phi.shape)
        for i in range(self.num_data):
            state = next_states[i]
            next_action = self.pick_epsilon_greedy_action(state, policy_weights, self.explore_epsilon)
            s0_ind = int(trans_data[i, 0])
            s1_ind = int(trans_data[i, 3])
            phi_s0 = old_Phi[s0_ind]
            phi_s1 = self.get_feature(state, next_action)
            new_Phi[s0_ind] = phi_s0
            new_Phi[s1_ind] = phi_s1
            A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
        A /= self.num_data
        return A, new_Phi

    def scale_data(self, trans_data, phi):
        A, b, C = MountainCar.init_abc(self.nFeatures)
        # self.phi_mean = np.mean(phi,axis=0)
        # self.phi_std = np.std(phi,axis=0)
        # phi = np.divide(phi-self.phi_mean, self.phi_std)
        self.phi_min = np.min(phi,axis=0)
        self.phi_max_min_diff = np.max(phi,axis=0) - self.phi_min
        phi = np.divide(phi - self.phi_min, self.phi_max_min_diff)
        for trans_data_t in trans_data:
            s_t = trans_data_t[0]
            s_t_1 = trans_data_t[3]
            reward = trans_data_t[2]
            phi_s0 = phi[int(s_t), :]
            phi_s1 = phi[int(s_t_1), :]
            A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
            C += np.outer(phi_s0, phi_s0)
            b += reward * phi_s0
        A /= self.num_data
        b /= self.num_data
        C /= self.num_data
        C_inv = self.compute_C_inv(C)
        return A,b,C,C_inv, phi

    def generate_phi(self):
        num_data_collected = 0
        num_states_visited = 0
        Phi = np.zeros((self.num_data * 2, self.nFeatures))
        next_states = np.zeros((self.num_data, self.state_dim))
        trans_data = np.zeros((self.num_data, 4))
        s0 = self.reset_env()
        action = self.env.action_space.sample()
        phi_s0 = self.get_feature(s0, action)
        while num_data_collected < self.num_data:
            s1, reward, done, _ = self.take_step_in_env(action)
            reward = self.reward_function(reward, s1, action, done)
            next_action = self.env.action_space.sample()
            phi_s1 = self.get_feature(s1, next_action)
            next_states[num_data_collected] = s1
            trans_data[num_data_collected] = np.array([num_states_visited, 0, reward, num_states_visited + 1])
            Phi[num_states_visited, :] = phi_s0
            Phi[num_states_visited + 1, :] = phi_s1
            phi_s0 = phi_s1
            action = next_action
            num_data_collected += 1
            num_states_visited += 2
            if done:
                s0 = self.reset_env()
                action = self.env.action_space.sample()
                phi_s0 = self.get_feature(s0, action)
        self.env.close()
        return Phi, trans_data, next_states

    def run_lspi_with_lstd(self):
        #self.saving_data_path = self.saving_data_path + 'lstd/'
        #A, b, C, C_inv, trans_data, Phi, next_states,_,_ = self.load_data() if self.is_loading else self.generate_data()
        # self.phi_min = np.load(self.loading_path + '/phi_min.npy')
        # self.phi_max_min_diff = np.load(self.loading_path + '/phi_max_min_diff.npy')

        # Phi, trans_data, next_states = self.generate_phi()
        # A,b,C,C_inv, Phi = self.scale_data(trans_data, Phi)

        A,b, _, _, _, _, _ = self.generate_data(saving_path=None, save_phi=False, save_trans_data=False, save_next_states=False, policy_weights=None)
        if matrix_rank(A) < self.nFeatures:
            print('A not full rank')
            A = A + 1e-5 * np.identity(self.nFeatures)
            print('rank of A = ' + str(matrix_rank(A)))
        # if matrix_rank(C) < self.nFeatures:
        #     print('C not full rank')
        #     C = C + 1e-5 * np.identity(self.nFeatures)
        #     print('min eig val of C = ', np.linalg.eig(C)[0][-1])
        policy_weights = np.linalg.solve(A, np.identity(self.nFeatures)) @ b
        #policy_weights = self.get_lstd_soln(A,b,C_inv,self.nFeatures)
        #cost_to_go_pole = self.get_cost_to_go(policy_weights)
        policy_score = self.get_policy_score(policy_weights)
        self.policy_scores = [policy_score]
        print('initial policy score is ' + str(policy_score))
        # if self.is_saving: self.save_data_to_cache(self.saving_data_path,
        #                                            {'A': A, 'b': b, 'C': C, 'Trans_data': trans_data, 'Phi': Phi,
        #                                             'C_inv': C_inv, 'next_states': next_states, 'phi_min':self.phi_min, 'phi_max_min_diff':self.phi_max_min_diff})
        for i in range(self.policy_iteration_episode):
            A, Phi = self.run_lstdq(policy_weights=policy_weights, trans_data=trans_data, next_states=next_states, old_Phi=Phi)
            if matrix_rank(A) < self.nFeatures:
                print('A not full rank')
                A = A + 1e-5*np.identity(self.nFeatures)
                print('rank of A = ' + str(matrix_rank(A)))
            policy_weights = np.linalg.solve(A, np.identity(self.nFeatures)) @ b
            #policy_weights = self.get_lstd_soln(A,b,C_inv,self.nFeatures)
            policy_score = self.get_policy_score(policy_weights)
            self.policy_scores.append(policy_score)
            print('policy_score is ' + str(policy_score))
            #cost_to_go_pole, cost_to_go_cart = self.get_cost_to_go(policy_weights)
        return [{'method_id':'lstd', 'policy_scores':self.policy_scores}]

    def get_feature(self, state, action):
        if self.feature_type == 'raw':
            return self.get_raw_feature(state, action)
        elif self.feature_type == 'perf':
            return self.get_perf_feature(state, action)
        elif self.feature_type == 'rbf':
            return self.get_rbf_feature(state, action)
        elif self.feature_type == 'approx':
            return self.get_approx_feature(state, action)
        elif self.feature_type == 'gray_down_sampled':
            return self.get_gray_down_sampled_feature(state, action)
        elif self.feature_type == 'vae':
            return self.get_vae_feature(state, action)

    def init_rbf_feature(self, num_rbf_means, state_means, rbf_sigma, state_min, state_max_min_diff, is_scaling_feature, num_state_rbf, is_normalizing_rbf_feature):
        self.num_rbf_means = num_rbf_means
        self.num_state_rbf = num_state_rbf
        self.state_means = state_means
        self.rbf_sigma = rbf_sigma
        self.state_min = state_min
        self.state_max_min_diff = state_max_min_diff
        if self.add_const_one_to_feature:
            self.num_feature_per_action = num_rbf_means ** self.num_state_rbf + 1
        else:
            self.num_feature_per_action = num_rbf_means ** self.num_state_rbf
        self.nFeatures = self.num_feature_per_action * self.num_actions
        self.is_scaling_feature = is_scaling_feature
        self.is_normalizing_rbf_feature = is_normalizing_rbf_feature

    def get_rbf_feature(self, state, action):
        if self.is_scaling_feature: state = np.divide(state-self.state_min, self.state_max_min_diff)
        feature = 0
        for i in range(self.num_state_rbf):
            feature += np.square(state[i]-self.state_means[i])
        feature = np.exp(-feature / self.rbf_sigma).flatten()
        if self.is_normalizing_rbf_feature: feature = np.divide(feature, np.sum(feature))
        if self.add_const_one_to_feature: feature = np.append(feature, 1)
        rbf_feature = np.zeros(self.nFeatures)
        rbf_feature[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = feature
        return rbf_feature

    def load_and_save_data_for_all_settings(self):
        #policy_iteration.load_mdp_info(self, self.loading_path+'/mdp_info.pkl')
        if self.is_loading: A, b, C, C_inv, trans_data, Phi, next_states, init_theta, init_omega = self.load_data()
        for exp_setting in self.exp_setup.exp_settings:
            method_id = self.get_method_id_from_dict(exp_setting)
            saving_path = os.path.join(self.saving_data_path, method_id)
            if self.is_loading == False: A, b, C, C_inv, init_theta, init_omega, file_handler = self.generate_data(saving_path=saving_path, save_phi=True, save_trans_data=True, save_next_states=False, policy_weights=None)
            if self.add_small_identity_to_A_C:
                A = A + 1e-5*np.identity(self.nFeatures)
                C = C + 1e-5*np.identity(self.nFeatures)
            self.save_abc_c_inv_h5py(file_handler, A,b,C,C_inv)
            policy_iteration.save_mdp_info(self, saving_path, {'num_data':self.num_data, 'nFeatures':self.nFeatures, 'gamma':self.gamma, 'feature_type':self.feature_type, 'num_feature_per_action':self.num_feature_per_action})
            policy_iteration.save_data_from_dict(self, saving_path, {'init_theta': init_theta, 'init_omega': init_omega})

    def load_and_save_large_data_for_all_settings(self):
        #Changed some code to only generate data once and copy the data file afterwards. This is only for large data experiment because generating data takes a lot of time
        generated_data = False
        data_file_path = None
        for exp_setting in self.exp_setup.exp_settings:
            method_id = self.get_method_id_from_dict(exp_setting)
            saving_path = os.path.join(self.saving_data_path, method_id)
            if generated_data: shutil.copy(os.path.join(data_file_path, 'data.hdf5'), saving_path)
            if self.is_loading == False and generated_data==False:
                A, b, C, C_inv, init_theta, init_omega, file_handler = self.generate_data(saving_path=saving_path, save_phi=True, save_trans_data=True, save_next_states=False, policy_weights=None)
                if self.add_small_identity_to_A_C:
                    A = A + 1e-5*np.identity(self.nFeatures)
                    C = C + 1e-5*np.identity(self.nFeatures)
                self.save_abc_c_inv_h5py(file_handler, A,b,C,C_inv)
                generated_data = True
                data_file_path = saving_path
            policy_iteration.save_mdp_info(self, saving_path, {'num_data':self.num_data, 'nFeatures':self.nFeatures, 'gamma':self.gamma, 'feature_type':self.feature_type, 'num_feature_per_action':self.num_feature_per_action})
            policy_iteration.save_data_from_dict(self, saving_path, {'init_theta': init_theta, 'init_omega': init_omega})

    def get_lstd_soln(self, exp_settings):
        data_path = os.path.join(self.saving_data_path, self.get_method_id_from_dict(exp_settings[0]))
        f = h5py.File(os.path.join(data_path, 'data.hdf5'), 'r')
        A = f.get('A')[0]
        b = f.get('b')[0]
        policy_weights = np.linalg.solve(A, np.identity(self.nFeatures)) @ b
        policy_score = self.get_policy_score(policy_weights)
        lstd_soln = {}
        lstd_soln['method_id'] = 'lstd'
        lstd_soln['policy_scores'] = policy_score
        lstd_soln['policy_weights'] = policy_weights
        return lstd_soln

    def load_data(self):
        A = np.load(self.loading_path + '/A.npy')
        b = np.load(self.loading_path + '/b.npy')
        C = np.load(self.loading_path + '/C.npy')
        C_inv = np.load(self.loading_path + '/C_inv.npy')
        next_states = np.load(self.loading_path + '/next_states.npy')
        trans_data = np.load(self.loading_path + '/Trans_data.npy')
        Phi = np.load(self.loading_path + '/Phi.npy')
        init_theta, init_omega = self.get_init_theta_omega()
        return A, b, C, C_inv, trans_data, Phi, next_states, init_theta, init_omega

    def generate_data(self, saving_path=None, save_phi=False, save_trans_data=False, save_next_states=False, policy_weights=None):
        if save_phi or save_trans_data or save_next_states: f = h5py.File(os.path.join(saving_path,'data.hdf5'), 'w')
        A, b, C = self.generate_data_per_method_id(save_phi, save_trans_data, save_next_states, f, policy_weights=None)
        C_inv = self.compute_C_inv(C)
        init_theta, init_omega = self.get_init_theta_omega()
        return A, b, C, C_inv, init_theta, init_omega, f

    def save_abc_c_inv_h5py(self, file_handler, A,b,C,C_inv):
        A_h5py = file_handler.create_dataset('A', (1, self.nFeatures, self.nFeatures), dtype='float64')
        A_h5py[0] = A
        b_h5py = file_handler.create_dataset('b', (1, self.nFeatures), dtype='float64')
        b_h5py[0] = b
        C_h5py = file_handler.create_dataset('C', (1, self.nFeatures, self.nFeatures), dtype='float64')
        C_h5py[0] = C
        C_inv_h5py = file_handler.create_dataset('C_inv', (1, self.nFeatures, self.nFeatures), dtype='float64')
        C_inv_h5py[0] = C_inv
        file_handler.flush()
        file_handler.close()

    def get_init_theta_omega(self):
        # init_theta = np.load(self.loading_path + '/init_theta.npy')
        # init_omega = np.load(self.loading_path + '/init_omega.npy')
        if self.init_method == 'random':
            init_theta = np.random.rand(self.nFeatures) * self.init_theta_omega_multiplier
            init_omega = np.random.rand(self.nFeatures) * self.init_theta_omega_multiplier
        elif self.init_method == 'previous' or self.init_method == 'zero':
            init_theta = np.zeros(self.nFeatures)
            init_omega = np.zeros(self.nFeatures)
        else:
            raise ValueError('invalid option for init_method')
        return init_theta, init_omega

    def compute_C_inv(self, C):
        if matrix_rank(C) < self.nFeatures: C = C + np.identity(self.nFeatures) * 1e-5
        C_inv = RandomMDP.compute_C_inv(self, C)
        return C_inv

    def pick_epsilon_greedy_action(self, state, policy_weights, explore_epsilon):
        if np.random.rand() < explore_epsilon:
            return np.random.choice(self.num_actions)
        else:
            return self.pick_greedy_action(state, policy_weights)

    def pick_greedy_action(self, state, policy_weights):
        state_action_values = [np.dot(self.get_feature(state, a), policy_weights) for a in range(self.num_actions)]
        return state_action_values.index(max(state_action_values))

    def save_mdp_info(self, path, data):
        policy_iteration.save_mdp_info(self,path,data)

    # def generate_data(self):
    #     for exp_setting in self.exp_setup.exp_settings:
    #         method_id = self.get_method_id_from_dict(exp_setting)
    #         A,b,C,Phi,trans_data,next_states = self.generate_data_per_method_id()
    #         C_inv = self.compute_C_inv(C)
    #         if self.is_saving: self.save_data_to_cache(self.saving_data_path+method_id+'/', {'A':A, 'A_before_loop':A, 'b':b, 'C':C, 'Trans_data':trans_data, 'Phi':Phi, 'Phi_before_loop':Phi,                                                                         'C_inv':C_inv, 'next_states':next_states, 'init_theta':np.random.rand(self.nFeatures)*self.init_theta_omega_multiplier, 'init_omega':np.random.rand(self.nFeatures)*self.init_theta_omega_multiplier})

class gym_env_policy_iteration:
    def generate_data_with_one_pass(self, save_phi=False, save_trans_data=False, save_next_states=False, saving_file_handler=None, policy_weights=None):
        num_data_collected = 0
        num_states_visited = 0
        if save_phi: phi_dataset_pointer = saving_file_handler.create_dataset('phi', (self.num_data * 2, self.nFeatures), dtype='float64',
                                               chunks=(1, self.nFeatures), compression="gzip", compression_opts=4)
        if save_trans_data: trans_dataset_pointer = saving_file_handler.create_dataset('trans_data', (self.num_data, 4), dtype='float64', chunks=(1, 4),
                                                 compression="gzip", compression_opts=4)
        if save_next_states: next_states_dataset_pointer = saving_file_handler.create_dataset('next_states', (self.num_data, self.state_dim), dtype='float64',
                                                       chunks=(1, self.state_dim), compression="gzip", compression_opts=4)

        A, b, C = MountainCar.init_abc(self.nFeatures)
        s0 = self.reset_env()
        action = self.env.action_space.sample() if policy_weights is None else self.pick_epsilon_greedy_action(self.num_actions, s0, policy_weights, self.num_features_per_acion, self.num_features)
        phi_s0 = self.get_feature(s0, action)
        while num_data_collected < self.num_data:
            s1, reward, done, _ = self.take_step_in_env(action)
            reward = self.reward_function(reward, s1, action, done)
            next_action = self.env.action_space.sample() if policy_weights is None else self.pick_epsilon_greedy_action(self.num_actions, s1, policy_weights, self.num_features_per_acion, self.num_features)
            phi_s1 = self.get_feature(s1, next_action)

            A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
            C += np.outer(phi_s0, phi_s0)
            b += reward * phi_s0

            if save_next_states: next_states_dataset_pointer[num_data_collected] = s1
            if save_trans_data: trans_dataset_pointer[num_data_collected] = np.array([num_states_visited, 0, reward, num_states_visited + 1])
            if save_phi:
                phi_dataset_pointer[num_states_visited] = phi_s0
                phi_dataset_pointer[num_states_visited+1] = phi_s1

            phi_s0 = phi_s1
            action = next_action
            num_data_collected += 1
            num_states_visited += 2
            if done:
                s0 = self.reset_env()
                action = self.env.action_space.sample() if policy_weights is None else self.pick_epsilon_greedy_action(self.num_actions, s0, policy_weights, self.num_features_per_acion, self.num_features)
                phi_s0 = self.get_feature(s0, action)
        self.env.close()
        A /= self.num_data
        b /= self.num_data
        C /= self.num_data

        return A, b, C

    def get_policy_score(self, policy_weights):
        num_trails = 100
        total_steps_to_terminate = 0
        for trail in range(num_trails):
            num_steps_per_trail = 0
            state = self.reset_env()

            while True:
                action = self.pick_greedy_action(state, policy_weights)
                state, reward, done, _ = self.take_step_in_env(action)
                num_steps_per_trail += 1
                #self.env.render()
                if done:
                    total_steps_to_terminate += num_steps_per_trail
                    break
        return total_steps_to_terminate / num_trails

class mountain_car_policy_iteration(continuous_env_policy_iteration, gym_env_policy_iteration):
    POSITION_MIN = -1.2
    POSITION_MAX = 0.5
    VELOCITY_MIN = -0.07
    VELOCITY_MAX = 0.07
    def __init__(self, exp_setup, feature_type, num_rbf_means, num_data, nFeatures, gamma, rbf_sigma, num_tilings, saving_path, is_saving, loading_path, is_loading, policy_iteration_episode, init_method, is_normalizing_rbf_feature, add_small_identity_to_A_C=False):
        continuous_env_policy_iteration.__init__(self, feature_type, num_rbf_means, rbf_sigma, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, add_const_one_to_feature=False, state_dim=2, init_method=init_method, add_small_identity_to_A_C=add_small_identity_to_A_C)
        self.env = gym.make('MountainCar-v0')
        self.init_theta_omega_multiplier = -20
        self.num_actions = 3
        if feature_type=='rbf':
            position_means, velocity_means = np.meshgrid(np.linspace(0, 1, num=self.num_rbf_means), np.linspace(0, 1, num=self.num_rbf_means))
            continuous_env_policy_iteration.init_rbf_feature(self, self.num_rbf_means, [position_means, velocity_means], rbf_sigma=rbf_sigma, state_min=np.array([self.POSITION_MIN,self.VELOCITY_MIN]), state_max_min_diff=np.array([self.POSITION_MAX-self.POSITION_MIN,self.VELOCITY_MAX-self.VELOCITY_MIN]), is_scaling_feature=True, num_state_rbf=2, is_normalizing_rbf_feature=is_normalizing_rbf_feature)
        else:
            self.num_tilings = num_tilings
            self.iht = IHT(nFeatures)
            self.positionScale = self.num_tilings / (self.POSITION_MAX - self.POSITION_MIN)
            self.velocityScale = self.num_tilings / (self.VELOCITY_MAX - self.VELOCITY_MIN)
            self.nFeatures = nFeatures

    def run_policy_iteration(self):
        return continuous_env_policy_iteration.run_lspi(self)

    def run_policy_iteration_with_lstd(self):
        return continuous_env_policy_iteration.run_lspi_with_lstd(self)

    def save_data_to_cache(self, path, data):
        info_dict = {'num_data': self.num_data, 'nFeatures': self.nFeatures, 'gamma': self.gamma, 'policy_iteration_episode': self.policy_iteration_episode}
        continuous_env_policy_iteration.save_mdp_info(self, path, info_dict)
        self.save_data_from_dict(path, data)

    def get_policy_score(self, policy_weights):
        return gym_env_policy_iteration.get_policy_score(self, policy_weights)
        # num_trails = 100
        # total_steps_to_terminate = 0
        # for trail in range(num_trails):
        #     num_steps_per_trail = 0
        #     cur_state = np.array([np.random.uniform(-0.6, -0.4), 0.0])
        #
        #     while True:
        #         action = self.pick_greedy_action(cur_state, policy_weights)
        #         next_state, _ = MountainCar.take_action_with_state(cur_state, action)
        #         num_steps_per_trail += 1
        #         cur_state = next_state
        #
        #         if cur_state[0] == self.POSITION_MAX:
        #             total_steps_to_terminate += num_steps_per_trail
        #             break
        #         elif num_steps_per_trail > 10000:
        #             return '> 10000'
        # return total_steps_to_terminate/num_trails

    def get_cost_to_go(self, policy_weights):
        num_sample_pts = 50
        positions, velocities = np.meshgrid(np.linspace(self.POSITION_MIN, self.POSITION_MAX, num=num_sample_pts), np.linspace(self.VELOCITY_MIN, self.VELOCITY_MAX, num=num_sample_pts))
        cost_to_go = np.zeros((num_sample_pts, num_sample_pts))
        action_choices = np.zeros((num_sample_pts, num_sample_pts))
        cost_to_go_down = np.zeros((num_sample_pts, num_sample_pts))
        cost_to_go_zero = np.zeros((num_sample_pts, num_sample_pts))
        cost_to_go_up = np.zeros((num_sample_pts, num_sample_pts))
        for i in range(num_sample_pts):
            for j in range(num_sample_pts):
                state_action_value = [np.dot(self.get_feature(positions[i][j], velocities[i][j], a), policy_weights) for a in range(3)]
                cost_to_go[i][j] = max(state_action_value)
                action_choices[i][j] = state_action_value.index(max(state_action_value))
                cost_to_go_down[i][j] = np.dot(self.get_feature(positions[i][j], velocities[i][j], 0), policy_weights)
                cost_to_go_zero[i][j] = np.dot(self.get_feature(positions[i][j], velocities[i][j], 1), policy_weights)
                cost_to_go_up[i][j] = np.dot(self.get_feature(positions[i][j], velocities[i][j], 2), policy_weights)
        return cost_to_go

    def generate_data_per_method_id(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights):
        return gym_env_policy_iteration.generate_data_with_one_pass(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights)
        # A, b, C = MountainCar.init_abc(self.nFeatures)
        # num_data_collected = 0
        # num_steps_per_episode = 0
        # num_states_visited = 0
        # num_trajs = 0
        # cur_state = np.array([np.random.uniform(-0.6, -0.4), 0.0])
        # cur_action = np.random.choice(3)
        # phi_s0 = self.get_feature(cur_state,cur_action)
        # # Phi = np.zeros((self.num_data * 2, self.nFeatures))
        # # next_states = np.zeros((self.num_data, self.state_dim))
        # # trans_data = np.zeros((self.num_data, 4))
        # while True:
        #     next_state, reward = MountainCar.take_action_with_state(cur_state, cur_action)
        #     nex_action = np.random.choice(3)
        #     phi_s1 = self.get_feature(next_state, nex_action)
        #
        #     A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
        #     C += np.outer(phi_s0, phi_s0)
        #     b += reward * phi_s0
        #
        #     # next_states[num_data_collected,:] = next_state
        #     # trans_data[num_data_collected] = np.array([num_states_visited,0,reward,num_states_visited+1])
        #     # Phi[num_states_visited, :] = phi_s0
        #     # Phi[num_states_visited+1, :] = phi_s1
        #
        #     phi_s0 = phi_s1
        #     cur_state = next_state
        #     cur_action = nex_action
        #
        #     num_steps_per_episode += 1
        #     num_data_collected += 1
        #     num_states_visited += 2
        #     if cur_state[0] == self.POSITION_MAX:
        #         cur_state, cur_action, phi_s0 = self.restart_in_data_generation()
        #         num_trajs += 1
        #         num_steps_per_episode = 0
        #         print('finished one episode at step=' + str(num_steps_per_episode) + '. finished ' + str(num_trajs) + ' trajectories.')
        #     elif num_steps_per_episode > 10000:
        #         cur_state, cur_action, phi_s0 = self.restart_in_data_generation()
        #         num_steps_per_episode = 0
        #         print('taking more than 10000 steps')
        #     elif num_data_collected == self.num_data:
        #         break
        # print('number of data collected = ' + str(num_data_collected))
        # A /= num_data_collected
        # b /= num_data_collected
        # C /= num_data_collected
        # #return A, b, C, Phi, trans_data, next_states
        # return A, b, C, 0, 0, 0

    def restart_in_data_generation(self):
        new_pos = np.random.uniform(-0.6, -0.4)
        new_velo = 0.0
        new_state = np.array([new_pos,new_velo])
        cur_action = np.random.choice(3)
        phi_s0 = self.get_feature(new_state, cur_action)
        return new_state, cur_action, phi_s0

    def take_step_in_env(self, action):
        return self.env.step(action)

    def reset_env(self):
        return self.env.reset()

    def reward_function(self, gym_reward, s0, action, is_done):
        return gym_reward

class Cart_Pole_Policy_Iteration(continuous_env_policy_iteration, gym_env_policy_iteration):
    def __init__(self, exp_setup, num_data=20000, gamma=0.95, feature_type='raw', num_rbf_means=3, rbf_sigma=0.5, add_const_one_to_feature=False, is_saving=False, saving_path=None, is_loading=True, loading_path=None, policy_iteration_episode=0, init_method='random', reward_type='gym', is_normalizing_rbf_feature=False, add_small_identity_to_A_C=False):
        self.reward_type = reward_type #Temporary
        continuous_env_policy_iteration.__init__(self, feature_type, num_rbf_means, rbf_sigma, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, add_const_one_to_feature, state_dim=4, init_method=init_method, add_small_identity_to_A_C=add_small_identity_to_A_C)
        self.env = gym.make('CartPole-v1')
        self.init_theta_omega_multiplier = -40
        self.num_actions = 2
        if feature_type == 'raw':
            self.num_feature_per_action = 4
        elif feature_type == 'approx':
            self.num_feature_per_action = 5
        elif feature_type == 'perf':
            self.num_feature_per_action = 81
            car_pos_poly, car_velo_poly, pole_angle_poly, pole_velo_poly = np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2])
            self.poly_kernel = np.array([car_pos_poly.flatten(), car_velo_poly.flatten(), pole_angle_poly.flatten(), pole_velo_poly.flatten()])
        elif feature_type == 'rbf':
            self.useful_state_inds = [0,1,2,3]
            cart_pos_means, cart_velo_means, pole_angle_means, pole_velo_means = np.meshgrid(np.array([-0.1,0,0.1]), np.array([-1,0,1]), np.array([-0.1,0,0.1]), np.array([-1,0,1]))
            continuous_env_policy_iteration.init_rbf_feature(self, num_rbf_means=3, state_means=[cart_pos_means, cart_velo_means, pole_angle_means, pole_velo_means], rbf_sigma=rbf_sigma, state_min=None, state_max_min_diff=None, is_scaling_feature=False, num_state_rbf=4, is_normalizing_rbf_feature=is_normalizing_rbf_feature)
            #self.useful_state_inds = [0,2,3]
            #cart_pos_means, pole_angle_means, pole_velo_means = np.meshgrid(np.array([-0.1,0,0.1]), np.array([-0.1,0,0.1]), np.array([-1,0,1]))
            #continuous_env_policy_iteration.init_rbf_feature(self, num_rbf_means=3, state_means=[cart_pos_means, pole_angle_means, pole_velo_means], rbf_sigma=rbf_sigma, state_min=None, state_max_min_diff=None, is_scaling_feature=False, num_state_rbf=3, is_normalizing_rbf_feature=is_normalizing_rbf_feature)
        self.nFeatures = self.num_feature_per_action * self.num_actions

    def run_policy_iteration(self):
        return continuous_env_policy_iteration.run_lspi(self)

    def run_policy_iteration_with_lstd(self):
        return continuous_env_policy_iteration.run_lspi_with_lstd(self)

    def generate_data_per_method_id(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights):
        #return self.generate_data_tracking_limits() if self.feature_type == 'rbf' else gym_env_policy_iteration.generate_data_with_one_pass(self)
        #return self.generate_data_tracking_limits()
        return gym_env_policy_iteration.generate_data_with_one_pass(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights)

    def generate_data_tracking_limits(self):
        num_data_collected = 0
        num_states_visited = 0
        A, b, C = MountainCar.init_abc(self.nFeatures)
        next_states = []
        trans_data = np.zeros((self.num_data, 4))
        raw_features = []
        s0 = self.env.reset()
        self.set_velo_limites(s0)
        while num_data_collected < self.num_data:
            action = self.env.action_space.sample()
            s1, reward, done, _ = self.env.step(action)
            self.update_velo_limits(s1)

            next_states.append(s1)
            trans_data[num_data_collected] = np.array([num_states_visited, action, reward, num_states_visited + 1])
            raw_features.append(s0)
            raw_features.append(s1)
            num_data_collected += 1
            num_states_visited += 2
            s0 = s1
            if done:
                s0 = self.env.reset()
        self.env.close()

        Phi = np.zeros((len(raw_features), self.nFeatures))
        cart_pos_means, cart_velo_means, pole_angle_means, pole_velo_means = np.meshgrid(np.linspace(0, 1, num=self.num_rbf_means), np.linspace(0, 1, num=self.num_rbf_means), np.linspace(0, 1, num=self.num_rbf_means), np.linspace(0, 1, num=self.num_rbf_means))
        continuous_env_policy_iteration.init_rbf_feature(self, self.num_rbf_means, [cart_pos_means,cart_velo_means,pole_angle_means,pole_velo_means], rbf_sigma=self.rbf_sigma, state_min=np.array([self.min_cart_pos, self.min_cart_velo, self.min_pole_angle, self.min_pole_velo]), state_max_min_diff=np.array([self.max_cart_pos-self.min_cart_pos, self.max_cart_velo - self.min_cart_velo, self.max_pole_angle-self.min_pole_angle,self.max_pole_velo - self.min_pole_velo]), is_scaling_feature=True, num_state_rbf=len(self.useful_state_inds), is_normalizing_rbf_feature=self.is_normalizing_rbf_feature)

        for i in range(self.num_data):
            trans_data_i = trans_data[i]
            s0 = int(trans_data_i[0])
            action = int(trans_data_i[1])
            reward = trans_data_i[2]
            s1 = int(trans_data_i[3])
            phi_s0 = self.get_feature(raw_features[s0], action)
            phi_s1 = self.get_feature(raw_features[s1], action)
            Phi[s0] = phi_s0
            Phi[s1] = phi_s1

            A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
            C += np.outer(phi_s0, phi_s0)
            b += reward * phi_s0

            A /= self.num_data
            b /= self.num_data
            C /= self.num_data
            Phi = np.array(Phi)
            next_states = np.array(next_states)
        return A, b, C, Phi, trans_data, next_states

    def get_raw_feature(self, state, action):
        # state[1] = np.tanh(state[1]/10)
        # state[3] = np.tanh(state[3]/10)
        phi = np.zeros(self.nFeatures)
        phi[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = np.array(state)
        return phi

    def get_perf_feature(self, state, action):
        # feature = [1]
        # for i in range(4):
        #     j = i
        #     while j < 4:
        #         feature.append(state[i]*state[j]) if i==j else feature.append(state[i]*state[j]*2)
        #         #feature.append(state[i]*state[j]) if i==j else feature.append(np.absolute(state[i]*state[j]*2))
        #         #feature.append(state[i]*state[j])
        #         j += 1
        num_state, num_poly_kernel = self.poly_kernel.shape
        feature = []
        for i in range(num_poly_kernel):
            feature.append(np.prod(np.power(state, self.poly_kernel[:,i])))
        #feature = self.build_poly_feature(state)
        phi = np.zeros(self.nFeatures)
        phi[action*self.num_feature_per_action:(action+1)*self.num_feature_per_action] = np.array(feature)
        #if hasattr(self, 'phi_mean') and hasattr(self, 'phi_std'): phi = np.divide(phi - self.phi_mean, self.phi_std)
        if hasattr(self, 'phi_min') and hasattr(self, 'phi_max_min_diff'): phi = np.divide(phi - self.phi_min, self.phi_max_min_diff)
        return phi

    def get_approx_feature(self, state, action):
        phi = np.zeros(self.nFeatures)
        phi[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = np.append(np.power(np.array(state),2), 1)
        return phi

    def get_rbf_feature(self, state, action):
        return continuous_env_policy_iteration.get_rbf_feature(self, state[self.useful_state_inds], action)

    def take_step_in_env(self, action):
        return self.env.step(action)

    def reset_env(self):
        return self.env.reset()

    def reward_function(self, gym_reward, state, action, is_done):
        if self.reward_type == 'gym':
            return gym_reward
        elif self.reward_type == 'lspi_paper':
            if is_done:
                return -1
            else:
                return 0
        elif self.reward_type == 'dann':
            return -100*state[0]**2 - state[1]**2 - state[2]**2 - 100*state[3]**2

    def set_velo_limites(self, init_state):
        self.max_cart_pos = init_state[0]
        self.min_cart_pos = init_state[0]
        self.max_cart_velo = init_state[1]
        self.min_cart_velo = init_state[1]
        self.max_pole_angle = init_state[2]
        self.min_pole_angle = init_state[2]
        self.max_pole_velo = init_state[3]
        self.min_pole_velo = init_state[3]

    def update_velo_limits(self, state):
        cur_cart_pos = state[0]
        cur_cart_velo = state[1]
        cur_pole_angle = state[2]
        cur_pole_velo = state[3]
        if cur_cart_velo > self.max_cart_velo:
            self.max_cart_velo = cur_cart_velo
        elif cur_cart_velo < self.min_cart_velo:
            self.min_cart_velo = cur_cart_velo
        if cur_pole_velo > self.max_pole_velo:
            self.max_pole_velo = cur_pole_velo
        elif cur_pole_velo < self.min_pole_velo:
            self.min_pole_velo = cur_pole_velo

        if cur_cart_pos > self.max_cart_pos:
            self.max_cart_pos = cur_cart_pos
        elif cur_cart_pos < self.min_cart_pos:
            self.min_cart_pos = cur_cart_pos
        if cur_pole_angle > self.max_pole_angle:
            self.max_pole_angle = cur_pole_angle
        elif cur_pole_angle < self.min_pole_angle:
            self.min_pole_angle = cur_pole_angle

    def get_policy_score(self, policy_weights):
        return gym_env_policy_iteration.get_policy_score(self, policy_weights)

    def get_cost_to_go(self, policy_weights):
        num_sample_pts = 50
        pole_angles, pole_velocities = np.meshgrid(np.linspace(-0.25, 0.25, num=40), np.linspace(-3, 3, num=num_sample_pts))
        cost_to_go_pole = np.zeros((num_sample_pts, 40))
        value_a0 = np.zeros((num_sample_pts, 40))
        value_a1 = np.zeros((num_sample_pts, 40))
        for i in range(num_sample_pts):
            for j in range(40):
                state = np.array([0,0,pole_angles[i][j], pole_velocities[i][j]])
                state_action_value = [np.dot(continuous_env_policy_iteration.get_feature(self,state,a), policy_weights) for a in range(self.num_actions)]
                #cost_to_go_pole[i][j] = max(state_action_value)
                #cost_to_go_pole[i][j] = state_action_value.index(max(state_action_value))
                value_a0[i][j] = state_action_value[0]
                value_a1[i][j] = state_action_value[1]

        # cart_positions, cart_velocities = np.meshgrid(np.linspace(-1.3, 1.3, num=num_sample_pts), np.linspace(-2.5, 2.5, num=num_sample_pts))
        # cost_to_go_cart = np.zeros((num_sample_pts, num_sample_pts))
        # for i in range(num_sample_pts):
        #     for j in range(num_sample_pts):
        #         state = np.array([cart_positions[i][j], cart_velocities[i][j], 0,0])
        #         state_action_value = [np.dot(continuous_env_policy_iteration.get_feature(self,state,a), policy_weights) for a in range(self.num_actions)]
        #         #cost_to_go_cart[i][j] = max(state_action_value)
        #         cost_to_go_cart[i][j] = state_action_value.index(max(state_action_value))

        return cost_to_go_pole

    def save_data_to_cache(self, path, data):
        if self.feature_type=='rbf':
            self.save_mdp_info(path, {'num_data': self.num_data, 'nFeatures': self.nFeatures, 'gamma': self.gamma,
                           'feature_type': self.feature_type, 'num_feature_per_action': self.num_feature_per_action})
                           #  ,'max_pole_velo': self.max_pole_velo, 'max_cart_velo': self.max_cart_velo,
                           # 'min_pole_velo': self.min_pole_velo, 'min_cart_velo': self.min_cart_velo})
        else:
            self.save_mdp_info(path, {'num_data': self.num_data, 'nFeatures': self.nFeatures, 'gamma': self.gamma,
                           'feature_type': self.feature_type, 'num_feature_per_action': self.num_feature_per_action})
        self.save_data_from_dict(path, data)

class Acrobot_Policy_Iteration(continuous_env_policy_iteration, gym_env_policy_iteration):
    def __init__(self, exp_setup, num_data=5000, gamma=0.95, feature_type='approx', num_rbf_means=3, rbf_sigma=1, add_const_one_to_feature=True, is_saving=False, saving_path=None, is_loading=True, loading_path=None, policy_iteration_episode=0, init_method='random', is_normalizing_rbf_feature=False, add_small_identity_to_A_C=False):
        continuous_env_policy_iteration.__init__(self, feature_type, num_rbf_means, rbf_sigma, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, add_const_one_to_feature, state_dim=4, init_method=init_method, add_small_identity_to_A_C=add_small_identity_to_A_C)
        self.env = gym.make('Acrobot-v1')
        self.init_theta_omega_multiplier = 10
        self.num_actions = 3
        if feature_type == 'rbf':
            self.useful_state_inds = np.array([0,2,4,5])
            theta_1_means, theta_2_means, theta_1_dot_means, theta_2_dot_means = np.meshgrid(np.linspace(0, 1, num=self.num_rbf_means),np.linspace(0, 1, num=self.num_rbf_means),np.linspace(0, 1, num=self.num_rbf_means),np.linspace(0, 1, num=self.num_rbf_means))
            continuous_env_policy_iteration.init_rbf_feature(self, self.num_rbf_means, [theta_1_means, theta_2_means, theta_1_dot_means, theta_2_dot_means], rbf_sigma=rbf_sigma, state_min=self.env.observation_space.low[self.useful_state_inds], state_max_min_diff=self.env.observation_space.high[self.useful_state_inds]-self.env.observation_space.low[self.useful_state_inds], is_scaling_feature=True, num_state_rbf=len(self.useful_state_inds), is_normalizing_rbf_feature=is_normalizing_rbf_feature)
        elif feature_type == 'approx':
            self.num_feature_per_action = 5
            self.nFeatures = self.num_feature_per_action * self.num_actions

    def run_policy_iteration(self):
        return continuous_env_policy_iteration.run_lspi(self)

    def run_policy_iteration_with_lstd(self):
        return continuous_env_policy_iteration.run_lspi_with_lstd(self)

    def generate_data_per_method_id(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights):
        return gym_env_policy_iteration.generate_data_with_one_pass(self, save_phi, save_trans_data, save_next_states, saving_file_handler, policy_weights)

    # def get_rbf_feature(self, state, action):
    #     return continuous_env_policy_iteration.get_rbf_feature(self,state[self.useful_state_inds],action)

    def get_approx_feature(self, state, action):
        phi = np.zeros(self.nFeatures)
        phi[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = np.square(np.array([np.arcsin(state[1]), np.arcsin(state[3]), state[4], state[5], 1]))
        return phi

    def take_step_in_env(self, action):
        s1, reward, done, obs = self.env.step(action)
        return s1[self.useful_state_inds], reward, done, obs

    def reset_env(self):
        return self.env.reset()[self.useful_state_inds]

    def get_policy_score(self, policy_weights):
        return gym_env_policy_iteration.get_policy_score(self, policy_weights)

    def save_data_to_cache(self, path, data):
        self.save_mdp_info(path, {'num_data': self.num_data, 'nFeatures': self.nFeatures, 'gamma': self.gamma,
                                  'feature_type': self.feature_type, 'num_feature_per_action': self.num_feature_per_action})
        self.save_data_from_dict(path, data)

    def reward_function(self, gym_reward, s0, action, is_done):
        return gym_reward

class RandomMDP_Policy_Iteration(policy_iteration):
    def __init__(self, exp_setup, num_data=20000, nStates=400, nActions=10, nFeatures=200, reward_level=1, feature_level=1, gamma=0.95, rho=0, num_states_reachable_from_sa=5, saving_path=None, is_saving=False, loading_path=None, is_loading=False, policy_iteration_episode=0, init_method='random'):
        policy_iteration.__init__(self, exp_setup, policy_iteration_episode, num_data, gamma, is_saving, saving_path, is_loading, loading_path, init_method)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_cached_data(loading_path) if is_loading else self.init_mdp_env(num_data,nStates,nActions,nFeatures,reward_level,feature_level,gamma,rho,num_states_reachable_from_sa,is_saving)

    def init_mdp_env(self, num_data, nStates, nActions, nFeatures, reward_level, feature_level, gamma, rho, num_states_reachable_from_sa, is_saving):
        self.num_data = num_data
        self.nStates = nStates
        self.nActions = nActions
        self.nFeatures = nFeatures
        self.reward_level = reward_level
        self.feature_level = feature_level
        self.gamma = gamma
        self.rho = rho
        self.is_saving = is_saving

        self.policy = np.random.rand(self.nStates, self.nActions)
        self.policy = self.policy / self.policy.sum(axis=1)[:,None]

        #self.rewards = -reward_level + 2*reward_level*np.random.rand(nStates, nActions, nStates) #used in thesis
        #self.rewards = reward_level*np.random.rand(nStates, nActions, nStates)
        #Used in Dann et al
        self.rewards = np.ones((nStates, nActions, nStates)) * reward_level
        self.rewards *= np.random.rand(nActions, nStates)[None, :, :]

        self.Phi = feature_level * np.random.rand(nStates, nFeatures)
        self.Phi = np.concatenate((self.Phi, np.ones((nStates, 1))), axis=1)
        self.nFeatures = self.Phi.shape[1]

        self.init_probs = np.random.rand(nStates) + 1e-5
        self.init_probs = self.init_probs / np.sum(self.init_probs)

        #used in thesis
        # self.trans_probs = np.zeros((nStates, nActions, nStates))
        # for s in range(nStates):
        #     for a in range(nActions):
        #         states_reachable = np.random.choice(nStates, size=num_states_reachable_from_sa)
        #         probs = np.random.rand(num_states_reachable_from_sa)
        #         probs = probs / np.sum(probs)
        #         self.trans_probs[s,a,states_reachable] = probs
        # self.trans_probs = self.trans_probs + 1e-5

        #used in Dann et al
        self.trans_probs = np.random.rand(nStates, nActions, nStates) + 1e-5

        self.trans_probs = self.trans_probs / self.trans_probs.sum(axis=2)[:, :, np.newaxis]
        assert np.all(self.trans_probs >= 0)
        assert np.all(self.trans_probs <= 1)
        self.s_terminal = np.asarray([np.all(self.trans_probs[s, :, s] == 1) for s in range(nStates)])
        self.init_testing_states = np.random.choice(self.nStates, 100, replace=False)
        self.optimal_policy, _ = self.value_iteration(termination_epsilon=1e-10)

    def init_policies(self):
        self.policies = {}
        self.policy_scores = {}
        self.policy_compute_costs = {}
        self.all_policy_eval_results = {}
        self.diff_with_optimal = {}
        init_policy = self.policy
        init_policy_score = self.get_policy_score(init_policy)
        print('initial policy score is %.5f' % (init_policy_score))
        init_policy_diff_with_optimal = self.get_policy_change(self.optimal_policy, init_policy)
        print('Difference between optimal and initial policy is %.7f' % (init_policy_diff_with_optimal))
        for exp_setting in self.exp_setup.exp_settings:
            method_id = self.get_method_id_from_dict(exp_setting)
            self.policies[method_id] = init_policy
            self.policy_scores[method_id] = [init_policy_score]
            self.policy_compute_costs[method_id] = [0]
            self.all_policy_eval_results[method_id] = []
            self.diff_with_optimal[method_id] = [init_policy_diff_with_optimal]

    # generate data under the current policy, evaluate and improve
    def run_policy_iteration(self):
        self.create_exp_saving_dir()
        self.init_policies()
        for i in range(self.policy_iteration_episode):
            print('\n\nepoch ' + str(i) + ' of policy iteration')
            self.generate_data()
            policy_eval_results = self.exp_setup.run_exps(self.exp_setup.exp_settings)
            self.improve_policy(policy_eval_results)
        return_vals = []
        for exp_setting in self.exp_setup.exp_settings:
            temp = {}
            method_id = self.get_method_id_from_dict(exp_setting)
            temp['method_id'] = method_id
            temp['policy_scores'] = self.policy_scores[method_id]
            temp['comp_costs'] = self.policy_compute_costs[method_id]
            temp['pol_eval_results'] = self.all_policy_eval_results[method_id]
            temp['diff_with_optimal'] = self.diff_with_optimal[method_id]
            return_vals.append(temp)
        return return_vals

    def load_cached_data(self, cached_data_path):
        self.load_mdp_info(cached_data_path + '/mdp_info.pkl')
        self.trans_probs = np.load(cached_data_path+'/trans_prob.npy')
        self.init_probs = np.load(cached_data_path+'/init_prob.npy')
        self.policy = np.load(cached_data_path+'/init_policy.npy')
        self.init_testing_states = np.load(cached_data_path+'/init_testing_states.npy')
        self.s_terminal = np.load(cached_data_path+'/terminal_states.npy')
        self.rewards = np.load(cached_data_path+'/rewards.npy')
        self.Phi = np.load(cached_data_path+'/Phi.npy')
        self.optimal_policy = np.load(cached_data_path+'/optimal_policy.npy')

    def save_data(self, path):
        policy_iteration.save_mdp_info(self, path, {'num_data': self.num_data, 'nStates': self.nStates, 'nActions': self.nActions, 'nFeatures': self.nFeatures, 'gamma': self.gamma, 'reward_level': self.reward_level, 'feature_level': self.feature_level, 'policy_iteration_episode':self.policy_iteration_episode})
        policy_iteration.save_data_from_dict(self, path, {'trans_prob':self.trans_probs, 'init_prob':self.init_probs, 'init_policy':self.policy, 'terminal_states':self.s_terminal, 'rewards':self.rewards, 'Phi':self.Phi, 'optimal_policy':self.optimal_policy, 'init_testing_states':self.init_testing_states})

    def get_exp_method_name(self, method):
        return method.__name__

    def generate_data(self):
        for exp_setting in self.exp_setup.exp_settings:
            method_id = self.get_method_id_from_dict(exp_setting)
            policy = policy_iteration.get_policy(self, method_id)
            A,b,C,trans_data,Phi,C_inv = self.generate_data_per_method_id(policy)
            policy_iteration.save_mdp_info(self, self.saving_data_path+method_id, {'num_data': self.num_data, 'nStates': self.nStates,
                                                        'nActions': self.nActions, 'nFeatures': self.nFeatures,
                                                        'gamma': self.gamma, 'reward_level': self.reward_level,
                                                        'feature_level': self.feature_level,
                                                        'policy_iteration_episode': self.policy_iteration_episode})
            policy_iteration.save_data_from_dict(self, self.saving_data_path+method_id+'/', {'A':A, 'b':b, 'C':C, 'Trans_data':trans_data, 'Phi':Phi, 'C_inv':C_inv})

    def generate_data_per_method_id(self, policy):
        A, b, C, C_inv, trans_data = self.get_abc_c_inv(policy)
        return A,b,C,trans_data,self.Phi,C_inv

    def improve_policy(self, policy_eval_results):
        Phi = self.Phi
        gamma = self.gamma
        nStates = self.nStates
        nActions = self.nActions
        for policy_eval_result in policy_eval_results:
            method_id = self.get_method_id_from_dict(policy_eval_result)
            new_policy = np.zeros((nStates, nActions))
            old_policy = policy_iteration.get_policy(self, method_id)

            #policy improvement
            value_function = Phi @ policy_eval_result['theta']
            for s in range(nStates):
                new_policy[s, :] = self.set_new_policy_for_s(s, gamma, nStates, nActions, value_function, old_policy)

            policy_score = self.get_policy_score(new_policy)
            policy_diff_with_optimal = self.get_policy_change(self.optimal_policy, new_policy)
            # print(method_id + ' policy_score is %.5f'%(policy_score))
            # print(method_id + ' change between policies is: %.7f' % (self.get_policy_change(old_policy, new_policy)))
            # print(method_id + ' difference between optimal and new policy is %.7f' % (policy_diff_with_optimal))

            #save values and update
            self.policy_scores[method_id].append(policy_score)
            self.policy_compute_costs[method_id].append(policy_eval_result['comp_cost'])
            self.diff_with_optimal[method_id].append(policy_diff_with_optimal)
            self.all_policy_eval_results[method_id].append(policy_eval_result['result'])
            policy_iteration.update_policy(new_policy, method_id)

    def run_policy_iteration_with_lstd(self):
        if self.is_saving: self.save_data(self.saving_data_path)
        policy_score = self.get_policy_score(self.policy)
        self.policy_scores = [policy_score]
        print('initial policy score is %.5f' % (policy_score))
        init_policy_diff_with_optimal = self.get_policy_change(self.optimal_policy, self.policy)
        self.diff_with_optimal = [init_policy_diff_with_optimal]
        print('difference between optimal and initial policy is %.3f\n\n' % (init_policy_diff_with_optimal))

        #mspbe_values = []
        for i in range(self.policy_iteration_episode):
            A,b,C,C_inv,_ = self.get_abc_c_inv(self.policy)
            lstd_theta = self.get_lstd_soln(A, b, C_inv, self.nFeatures)
            self.policy = self.improve_policy_lstd(lstd_theta, self.policy)
            # temp = A@lstd_theta - b
            # mspbe_values.append(0.5*np.dot(temp, C_inv@temp))

            policy_score = self.get_policy_score(self.policy)
            self.policy_scores.append(policy_score)
            print('policy_score is %.5f' % (policy_score))
            policy_diff_with_optimal = self.get_policy_change(self.optimal_policy, self.policy)
            self.diff_with_optimal.append(policy_diff_with_optimal)
            print('difference between optimal and current policy is %.3f\n\n'%(policy_diff_with_optimal))
        return [{'method_id':'lstd', 'policy_scores':self.policy_scores,'diff_with_optimal':self.diff_with_optimal}]
        #return mspbe_values

    def improve_policy_lstd(self, lstd_theta, old_policy):
        Phi = self.Phi
        gamma = self.gamma
        nStates = self.nStates
        nActions = self.nActions

        new_policy = np.zeros((nStates, nActions))
        value_function = Phi @ lstd_theta
        for s in range(nStates):
            new_policy[s, :] = self.set_new_policy_for_s(s, gamma, nStates, nActions, value_function, old_policy)
        return new_policy

    def get_policy_change(self, old_policy, new_policy):
        return np.sum(np.absolute(old_policy.flatten() - new_policy.flatten()))

    def set_new_policy_for_s(self, s, gamma, nStates, nActions, value_function, old_policy):
        values_per_action = np.zeros(nActions)
        for a in range(nActions):
            values_per_action[a] = np.sum(np.multiply(self.trans_probs[s,a,:]*old_policy[s,a], self.rewards[s,a,:]+gamma*value_function))
        values_per_action = np.rint(values_per_action)
        best_actions_inds = np.argwhere(values_per_action == np.amax(values_per_action)).flatten()
        values_per_action = np.zeros(nActions)
        values_per_action[best_actions_inds] = 1/len(best_actions_inds)
        return values_per_action

    def update_policy(self, new_policy, method_id):
        self.policies[method_id] = new_policy

    def get_abc_c_inv(self, policy):
        A, b, C = RandomMDP.init_abc(self, self.nFeatures, self.device)
        A, b, C, trans_data = RandomMDP.generate_data(self, A, b, C, self.num_data, self.nStates,
                                                          self.nActions, self.gamma, self.Phi,
                                                          self.init_probs, self.trans_probs, self.rewards,
                                                          self.s_terminal, self.device, policy)
        C_inv = RandomMDP.compute_C_inv(self, C)
        return A,b,C,C_inv,trans_data

    def get_policy_score(self, policy):
        num_episode = 100
        max_step_per_episode = 200
        episode_rewards = []
        for i in range(num_episode):
            episode_reward = 0
            s_0 = self.init_testing_states[i]
            #s_0 = np.random.choice(self.nStates, p=self.init_probs)
            for j in range(max_step_per_episode):
                a = np.random.choice(self.nActions, p=policy[s_0, :])
                s_1 = np.random.choice(self.nStates, p=self.trans_probs[s_0,a,:])
                r = self.rewards[s_0, a, s_1]
                episode_reward += r
                s_0 = s_1
            episode_rewards.append(episode_reward/max_step_per_episode)
        return np.sum(np.array(episode_rewards))/num_episode

    def value_iteration(self, termination_epsilon):
        iteration = 0
        nStates = self.nStates
        nActions = self.nActions
        values = np.random.rand(nStates)
        while True:
            s = np.random.randint(nStates)
            action_values_at_s = np.zeros(nActions)
            for a in range(nActions):
                action_values_at_s[a] = np.sum(np.multiply(self.trans_probs[s,a,:], self.rewards[s,a,:]+self.gamma*values))
            new_value = np.amax(action_values_at_s)
            if np.absolute(new_value-values[s])<termination_epsilon:
                print('value iteration terminates in ' + str(iteration) + ' iterations')
                break
            else:
                values[s] = new_value
                iteration += 1

        #output policy from values
        policy = np.zeros((nStates, nActions))
        for s in range(nStates):
            values_per_action = np.zeros(nActions)
            for a in range(nActions):
                values_per_action[a] = np.sum(np.multiply(self.trans_probs[s,a, :], self.rewards[s,a, :]+self.gamma*values))
            values_per_action = np.rint(values_per_action)
            best_actions_inds = np.argwhere(values_per_action == np.amax(values_per_action)).flatten()
            policy[s, best_actions_inds] = 1/len(best_actions_inds)

        return policy, values