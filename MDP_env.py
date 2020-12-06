import os
import random
random.seed(1)
import gym
import torch
import numpy as np
import pandas as pd
import datetime
#from scipy.linalg import eigh
from numpy.linalg import matrix_rank
from numpy.linalg import norm
from numpy.linalg import pinv
import math

# class gym_env:
#     def __init__(self, env_name, num_data, policy_weights, num_features=200, gamma=0.95, num_tilings=8, space_max_thres=1000, cached_data_path=None, load_cache_data=True, saving_data_path=None, is_saving=False):
#         self.cached_data_path = './MDP_cache/'+env_name+'/' if cached_data_path is None and load_cache_data else cached_data_path
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         if load_cache_data:
#             if self.cached_data_path == None: raise ValueError('path to cached data cannot be None')
#             RandomMDP.load_cached_data(self, self.cached_data_path)
#         else:
#             self.init_mdp_env(num_data, num_features, gamma, num_tilings, policy_weights, env_name, space_max_thres)
#
#     def init_mdp_env(self, num_data, num_features, gamma, num_tilings, policy_weights, env_name, space_max_thres):
#         self.num_data = num_data
#         self.num_features = num_features
#         self.gamma = gamma
#         self.num_tilings = num_tilings
#
#         self.env = self.create_gym_env(env_name)
#         self.set_scale(space_max_thres)
#
#         self.init_abc()
#         self.generate_data(policy_weights)
#
#     #if observation or action space is unbounded, set it to space_max_thres
#     def set_scale(self, space_max_thres):
#         self.action_space = self.get_diff(self.env.action_space)
#         self.state_space = self.get_diff(self.env.state_space)
#
#     #get the difference of observation or action space's max and min values. Set to space_max_thres if difference is inf
#     def get_diff(self, space, space_max_thres):
#         diff = space.high - space.low
#         space[np.isinf(diff)] = space_max_thres
#         return space
#
#     def create_gym_env(self, env_name):
#         env = gym.make(env_name)
#         env = env.reset()
#         return env
#
#     def init_mdp_env(self, num_data, num_features, gamma, num_tilings):
#         self.num_data = num_data
#         self.nFeatures = num_features
#         self.gamma = gamma
#         self.num_tilings = num_tilings
#
#     def init_abc(self):
#         self.A = torch.zeros([self.nFeatures, self.nFeatures], dtype=torch.float32, device=self.device)
#         self.b = torch.zeros([self.nFeatures], dtype=torch.float32, device=self.device)
#         self.C = torch.zeros([self.nFeatures, self.nFeatures], dtype=torch.float32, device=self.device)
#
#     def generate_data(self, policy_weights):

# def estimate_memory_usage(self, nStates, nActions, nFeatures, num_data):
#     memory_usage = 0
#     memory_usage += nStates * nActions * 8 * 2  # policy array + feature array
#     memory_usage += nStates * nActions * nStates * 8 * 2  # rewards array + transition probabilities array
#     memory_usage += nStates * 8 * 2  # initial probabilities array + value function
#
#     memory_usage += nFeatures * 8 * 3  # old & new theta array + b array
#     memory_usage += nFeatures * nFeatures * 8 * 3  # A and C and C_inv matrix
#     memory_usage += num_data * 4 * 8  # trans_data array
#     return memory_usage * 1e-9  # return in gigabytes

class RandomMDP:
    def __init__(self, policy='random', num_data=20000, nStates=400, nActions=10, nFeatures=200, reward_level=1, feature_level=1,gamma=0.95, rho=0, saving_data_path=None, is_saving=False):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.init_mdp_env(policy, num_data, nStates, nActions, nFeatures, reward_level, feature_level, gamma, rho, is_saving)
        if is_saving:
            mdp_info = {'num_data': self.num_data, 'nStates': self.nStates, 'nActions': self.nActions,
                    'nFeatures': self.nFeatures, 'gamma': self.gamma, 'rho': self.rho, 'reward_level': self.reward_level,
                    'feature_level': self.feature_level}
            RandomMDP.save_data(saving_data_path, self.A, self.b, self.C, self.C_inv, self.trans_data, self.Phi, mdp_info)

    def init_abc(self, nFeatures, device):
        A = torch.zeros([nFeatures, nFeatures], dtype=torch.float64, device=device)
        b = torch.zeros([nFeatures], dtype=torch.float64, device=device)
        C = torch.zeros([nFeatures, nFeatures], dtype=torch.float64, device=device)
        return A, b, C

    def init_mdp_env(self,policy,num_data,nStates,nActions,nFeatures,reward_level,feature_level,gamma,rho,is_saving):
        self.num_data = num_data
        self.nStates = nStates
        self.nActions = nActions
        self.nFeatures = nFeatures
        self.gamma = gamma
        self.rho = rho
        self.reward_level = reward_level
        self.feature_level = feature_level
        self.is_saving = is_saving
        # self.rewards = np.ones((nStates, nActions, nStates)) * reward_level
        # self.rewards *= np.random.rand(nActions, nStates)[None, :, :]
        self.rewards = -reward_level + 2 * reward_level * np.random.rand(nStates, nActions, nStates)

        self.policy = self.set_policy(policy)
        self.init_probs = np.random.rand(nStates) + 1e-5
        self.init_probs = self.init_probs / np.sum(self.init_probs)
        self.trans_probs = np.random.rand(nStates, nActions, nStates) + 1e-5
        self.trans_probs = self.trans_probs / self.trans_probs.sum(axis=2)[:, :, np.newaxis]
        assert np.all(self.trans_probs >= 0)
        assert np.all(self.trans_probs <= 1)
        self.s_terminal = np.asarray([np.all(self.trans_probs[s, :, s] == 1) for s in range(nStates)])

        #self.Phi = -feature_level + 2 * feature_level * np.random.rand(nStates, nFeatures)
        self.Phi = feature_level*np.random.rand(nStates, nFeatures)
        self.Phi = np.concatenate((self.Phi, np.ones((nStates, 1))), axis=1)
        self.nFeatures = self.Phi.shape[1]

        A, b, C = self.init_abc(self.nFeatures, self.device)
        self.A, self.b, self.C, self.trans_data = self.generate_data(A, b, C, num_data, nStates, nActions, gamma, self.Phi, self.init_probs,
                                                 self.trans_probs, self.rewards, self.s_terminal, self.device,
                                                 self.policy)
        self.C_inv = self.compute_C_inv(self.C)

    def compute_C_inv(self, C):
        nFeatures = C.shape[0]
        rank_c = matrix_rank(C)
        print('rank of C is ' + str(rank_c))
        C_inv = np.linalg.solve(C,np.identity(nFeatures)) if rank_c == nFeatures else np.linalg.lstsq(C, np.identity(nFeatures))
        return C_inv

    def set_policy(self, policy):
        if policy=='random':
            policy = np.random.rand(self.nStates, self.nActions)
            policy = policy / policy.sum(axis=1)[:,None]
        return policy

    def generate_data(self, A, b, C, num_data, nStates, nActions, gamma, Phi, init_probs, trans_probs, rewards, s_terminal, device, policy):
        # TODO make this a variable !!!!!!!!!!
        #max_data_per_episode = int(num_data/10) #used in thesis
        max_data_per_episode = 500
        trans_data = np.zeros((num_data, 4))
        i = 0
        Phi = torch.from_numpy(Phi).to(device)
        print('start generating' + str(datetime.datetime.now()))
        while i < num_data:
            ep_i = 0
            s_0 = np.random.choice(nStates, p=init_probs)
            while ep_i < max_data_per_episode:
                if s_terminal[s_0]: break
                a = np.random.choice(nActions, p=policy[s_0, :])
                s_1 = np.random.choice(nStates, p=trans_probs[s_0, a, :])
                r = rewards[s_0, a, s_1]
                trans_data[i, :] = np.array([s_0, a, r, s_1])

                phi_s0 = Phi[s_0, :]
                phi_s1 =  Phi[s_1, :]
                A_t = torch.ger(phi_s0, phi_s0 - gamma * phi_s1)
                C_t = torch.ger(phi_s0, phi_s0)
                A += A_t
                b += r * phi_s0
                C += C_t

                s_0 = s_1
                ep_i += 1
                i += 1
        print('finish generating' + str(datetime.datetime.now()))

        A /= num_data
        b /= num_data
        C /= num_data

        A = A.cpu().numpy()
        b = b.cpu().numpy()
        C = C.cpu().numpy()

        return A, b, C, trans_data

    @staticmethod
    def save_data(saving_data_path, A, b, C, C_inv, trans_data, Phi, mdp_info):
        if saving_data_path == None: raise ValueError('path to save data cannot be None')
        pd.DataFrame(data=mdp_info, index=[0]).to_pickle(saving_data_path+'mdp_info.pkl')
        np.save(saving_data_path+'A', A)
        np.save(saving_data_path+'b', b)
        np.save(saving_data_path+'C', C)
        np.save(saving_data_path+'C_inv', C_inv)
        np.save(saving_data_path+'Trans_data', trans_data)
        np.save(saving_data_path+'Phi', Phi)
        mdp_info_txt_fo = open(saving_data_path+'mdp_info.txt',"w")
        for k, v in mdp_info.items():
            mdp_info_txt_fo.write(str(k) + ' >>> ' + str(v) + '\n')
        mdp_info_txt_fo.close()

    def calc_lipschitz_const_for_t(self, A_t, C_t, rho):
        G = torch.cat((rho*torch.zeros([self.nFeatures,self.nFeatures],dtype=torch.float64, device=self.device), -torch.t(A_t)), dim=1)
        G = torch.cat((G, torch.cat((-A_t,-C_t),dim=1)), dim=0)
        G = torch.matmul(torch.t(G),G)
        eigvals = torch.symeig(G)
        return torch.sqrt(eigvals[0][2*self.nFeatures-1])

class MountainCar:
    POSITION_MIN = -1.2
    POSITION_MAX = 0.5
    VELOCITY_MIN = -0.07
    VELOCITY_MAX = 0.07

    INIT_VELO = 0
    ACTIONS = [-1,0,1]

    def __init__(self, num_data=20000, max_episodes=1200, num_features=100, num_features_sarsa=4096, gamma=0.95, num_tilings=8, saving_data_path=None, is_saving=False):
        self.init_mdp_env(num_data, max_episodes, num_features, gamma, num_tilings, num_features_sarsa)
        if is_saving: RandomMDP.save_data(saving_data_path, self.A, self.b, self.C, self.C_inv, self.trans_data, self.Phi, {'num_data':num_data,'num_features':num_features,'max_episodes':max_episodes,'gamma':gamma,'num_tilings':num_tilings})

    def init_abc(nFeatures):
        A = np.zeros((nFeatures, nFeatures))
        b = np.zeros(nFeatures)
        C = np.zeros((nFeatures, nFeatures))
        return A, b, C

    def init_mdp_env(self, num_data, max_episodes, num_features, gamma, num_tilings, num_features_sarsa):
        self.num_data = num_data
        self.nFeatures = num_features
        self.nFeatures_sarsa = num_features_sarsa
        self.gamma = gamma
        self.num_tilings = num_tilings

        #make it compatible with RandomMDP
        self.nStates = 0
        self.nActions = 0
        self.rho = 0
        self.reward_level = 0
        self.feature_level = 0

        print('learn policy using sarsa. start' + str(datetime.datetime.now()))
        policy_wgts = self.learn_policy(max_episodes)
        print('saving policy weights')
        np.save('./policy_weights.npy', policy_wgts)

        # policy_wgts = np.load('./policy_weights.npy')
        # self.iht = IHT(self.nFeatures)
        # self.positionScale = self.num_tilings / (self.POSITION_MAX - self.POSITION_MIN)
        # self.velocityScale = self.num_tilings / (self.VELOCITY_MAX - self.VELOCITY_MIN)

        iht_states = IHT(self.nFeatures)
        A,b,C = MountainCar.init_abc(self.nFeatures)
        self.A,self.b,self.C,self.trans_data,self.Phi = MountainCar.generate_data(policy_wgts, iht_states, self.iht_sarsa,
                                                         num_data, num_features, num_tilings, gamma,
                                                         A, b, C, self.positionScale, self.velocityScale)
        self.C_inv = RandomMDP.compute_C_inv(self, self.C)

    def generate_data(policy_wgts, iht_states, iht_sarsa, num_data, num_features, num_tilings, gamma, A, b, C, positionScale, velocityScale):
        trans_data = []
        Phi = []
        i = 0
        j = 0
        while j < num_data:
            finish_episode = False
            cur_pos = np.random.uniform(-0.6, -0.4)
            cur_velo = 0.0
            phi_s0 = MountainCar.get_feature_for_state(iht_states, cur_pos, cur_velo, num_tilings, num_features, positionScale, velocityScale)

            while j < num_data:
                action = MountainCar.pick_greedy_action(iht_sarsa, cur_pos, cur_velo, policy_wgts, num_tilings, positionScale, velocityScale)
                nex_pos, nex_velo, reward = MountainCar.take_action(cur_pos, cur_velo, action)
                phi_s1 = MountainCar.get_feature_for_state(iht_states, nex_pos, nex_velo, num_tilings, num_features, positionScale, velocityScale)

                A += np.outer(phi_s0, phi_s0-gamma*phi_s1)
                C += np.outer(phi_s0, phi_s0)
                b += reward * phi_s0
                trans_data.append(np.array([i,0,reward,i+1]))
                Phi.append(phi_s0)

                phi_s0 = phi_s1
                cur_pos = nex_pos
                cur_velo = nex_velo
                i += 1
                j += 1
                if cur_pos == MountainCar.POSITION_MAX:
                    Phi.append(phi_s1)
                    print('finished one episode at i=' + str(i))
                    i += 1
                    finish_episode = True
                    break
        if finish_episode == False: Phi.append(phi_s1)
        print('generated '+str(i))

        A /= num_data
        b /= num_data
        C /= num_data
        Phi = np.array(Phi)
        trans_data = np.array(trans_data)

        return A, b, C, trans_data, Phi

    def learn_policy(self, max_episodes):
        self.iht_sarsa = IHT(self.nFeatures_sarsa)
        self.positionScale = self.num_tilings / (self.POSITION_MAX - self.POSITION_MIN)
        self.velocityScale = self.num_tilings / (self.VELOCITY_MAX - self.VELOCITY_MIN)
        policy_wgts = MountainCar.semi_gradient_sarsa(max_episodes, self.iht_sarsa, self.nFeatures_sarsa, self.num_tilings, self.gamma, self.positionScale, self.velocityScale)
        return policy_wgts

    def take_action(position, velocity, action):
        action = MountainCar.ACTIONS[action]
        newVelocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
        newVelocity = min(max(MountainCar.VELOCITY_MIN, newVelocity), MountainCar.VELOCITY_MAX)
        newPosition = position + newVelocity
        newPosition = min(max(MountainCar.POSITION_MIN, newPosition), MountainCar.POSITION_MAX)
        reward = -1.0
        if newPosition == MountainCar.POSITION_MIN:
            newVelocity = 0.0
        return newPosition, newVelocity, reward

    def take_action_with_state(state, action):
        position = state[0]
        velocity = state[1]
        action = MountainCar.ACTIONS[action]
        newVelocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
        newVelocity = min(max(MountainCar.VELOCITY_MIN, newVelocity), MountainCar.VELOCITY_MAX)
        newPosition = position + newVelocity
        newPosition = min(max(MountainCar.POSITION_MIN, newPosition), MountainCar.POSITION_MAX)
        reward = -1.0
        if newPosition == MountainCar.POSITION_MIN:
            newVelocity = 0.0
        return np.array([newPosition, newVelocity]), reward

    def get_feature_for_state(iht_states, position, velocity, num_tilings, nFeatures, positionScale, velocityScale):
        feature = np.zeros(nFeatures)
        feature[tiles(iht_states, num_tilings, [positionScale * position, velocityScale * velocity])] = 1
        return feature

    def get_active_tiles(iht, position, velocity, action, num_tilings, positionScale, velocityScale):
        activeTiles = tiles(iht, num_tilings,
                            [positionScale * position, velocityScale * velocity],
                            [action])
        return activeTiles

    def get_value(iht, position, velocity, action, policy_wgts, num_tilings, positionScale, velocityScale):
        activeTiles = MountainCar.get_active_tiles(iht, position,velocity,action,num_tilings,positionScale,velocityScale)
        return np.sum(policy_wgts[activeTiles])

    def pick_epsilon_greedy_action(iht, position, velocity, policy_wgts, num_tilings, positionScale, velocityScale, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(3)
        else:
            state_action_values = [MountainCar.get_value(iht, position, velocity, a, policy_wgts, num_tilings, positionScale, velocityScale) for a in [0, 1, 2]]
            return state_action_values.index(max(state_action_values))

    def pick_greedy_action(iht, position, velocity, policy_wgts, num_tilings, positionScale, velocityScale):
        state_action_values = [MountainCar.get_value(iht, position, velocity, a, policy_wgts, num_tilings, positionScale, velocityScale) for a in [0, 1, 2]]
        return state_action_values.index(max(state_action_values))

    def semi_gradient_sarsa(max_episodes, iht_sarsa, nFeatures_sarsa, num_tilings, gamma, positionScale, velocityScale):
        policy_wgts = np.zeros(nFeatures_sarsa)
        LR = 0.01
        EPSILON = 0.1
        for i in range(max_episodes):
            j = 0
            cur_pos = np.random.uniform(-0.6, -0.4)
            cur_velo = 0.0
            action = MountainCar.pick_epsilon_greedy_action(iht_sarsa, cur_pos, cur_velo, policy_wgts, num_tilings, positionScale, velocityScale, EPSILON)

            while True:
                nex_pos, nex_velo, reward = MountainCar.take_action(cur_pos, cur_velo, action)
                active_tiles = MountainCar.get_active_tiles(iht_sarsa, cur_pos, cur_velo, action, positionScale, velocityScale)
                state_value = np.sum(policy_wgts[active_tiles])
                if nex_pos == MountainCar.POSITION_MAX:
                    policy_wgts[active_tiles] += LR * (reward-state_value)
                    print('episode '+ str(i) + ' Used ' + str(j) + ' steps.')
                    break
                action = MountainCar.pick_epsilon_greedy_action(iht_sarsa, nex_pos, nex_velo, policy_wgts, num_tilings, positionScale, velocityScale, EPSILON)
                nex_state_value = MountainCar.get_value(iht_sarsa, nex_pos, nex_velo, action, policy_wgts, num_tilings, positionScale, velocityScale)
                delta = reward + (gamma * nex_state_value) - state_value
                policy_wgts[active_tiles] += LR * delta
                cur_pos = nex_pos
                cur_velo = nex_velo
                j += 1
                if j > 1000000:
                    print('stuck for too long. Increase exploration rate and learning rate')
                    LR = 0.05
                    EPSILON = 0.5
            LR = max(LR*0.9, 0.01)
            EPSILON = max(EPSILON*0.9, 0.1)
        return policy_wgts



# class PoleBalancingMDP:
#     def __init__(self,num_data=20000, num_rbf_means=3, gamma=0.95, is_saving=False, saving_path=None, feature_type='raw'):
#             self.num_data = num_data
#             self.num_rbf_means = num_rbf_means
#             self.gamma = gamma
#             self.env = gym.make('CartPole-v1')
#             self.is_saving = is_saving
#             self.saving_path = saving_path
#             self.feature_type = feature_type
#             if feature_type == 'raw':
#                 self.num_feature_per_action = 4
#             elif feature_type == 'perf':
#                 self.num_feature_per_action = 11
#             elif feature_type == 'rbf':
#                 self.num_feature_per_action = 82
#                 self.cart_position_means, self.cart_velo_means, self.pole_angle_means, self.pole_velo_means = np.meshgrid(np.linspace(0, 1, num=num_rbf_means), np.linspace(0, 1, num=num_rbf_means), np.linspace(0, 1, num=num_rbf_means), np.linspace(0, 1, num=num_rbf_means))
#
#     def generate_data(self):
#         self.nFeatures = self.num_feature_per_action * 2
#
#         if self.feature_type == 'rbf':
#             A, b, C, Phi, trans_data, next_states = self.generate_data_tracking_limits()
#         else:
#             A, b, C, Phi, trans_data, next_states = self.generate_data_with_one_pass()
#
#         A /= self.num_data
#         b /= self.num_data
#         C /= self.num_data
#         Phi = np.array(Phi)
#         next_states = np.array(next_states)
#
#         if matrix_rank(C) < self.nFeatures: C = C + np.identity(self.nFeatures) * 1e-7
#         C_inv = RandomMDP.compute_C_inv(self, C)
#
#         if self.is_saving:
#             saving_data = {'num_data': self.num_data, 'nFeatures': self.nFeatures, 'gamma': self.gamma, 'feature_type':self.feature_type, 'num_feature_per_action':self.num_feature_per_action}
#             pd.DataFrame(data=saving_data, index=[0]).to_pickle(self.saving_path + '/mdp_info.pkl')
#             np.save(self.saving_path+'/A.npy', A)
#             np.save(self.saving_path+'/b.npy', b)
#             np.save(self.saving_path+'/C.npy', C)
#             np.save(self.saving_path+'/C_inv.npy', C_inv)
#             np.save(self.saving_path+'/next_states.npy', next_states)
#             np.save(self.saving_path + '/Phi.npy', Phi)
#             np.save(self.saving_path + '/Trans_data.npy', trans_data)
#
#     def generate_data_tracking_limits(self):
#         num_data_collected = 0
#         num_states_visited = 0
#         A, b, C = MountainCar.init_abc(self.nFeatures)
#         next_states = []
#         trans_data = np.zeros((self.num_data, 4))
#         raw_features = []
#         s0 = self.env.reset()
#         self.set_velo_limites(s0)
#         while num_data_collected < self.num_data:
#             action = self.env.action_space.sample()
#             s1, reward, done, _ = self.env.step(action)
#             self.update_velo_limits(s1)
#
#             next_states.append(s1)
#             trans_data[num_data_collected] = np.array([num_states_visited, action, reward, num_states_visited + 1])
#             raw_features.append(s0)
#             raw_features.append(s1)
#             num_data_collected += 1
#             num_states_visited += 2
#             if done:
#                 s0 = self.env.reset()
#         self.env.close()
#
#         Phi = np.zeros((len(raw_features), self.nFeatures))
#         self.set_state_min_and_diff()
#         for i in range(self.num_data):
#             trans_data_i = trans_data[i]
#             s0 = int(trans_data_i[0])
#             action = int(trans_data_i[1])
#             reward = trans_data_i[2]
#             s1 = int(trans_data_i[3])
#             phi_s0 = self.get_feature(raw_features[s0], action)
#             phi_s1 = self.get_feature(raw_features[s1], action)
#             Phi[s0] = phi_s0
#             Phi[s1] = phi_s1
#
#             A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
#             C += np.outer(phi_s0, phi_s0)
#             b += reward * phi_s0
#         return A, b, C, Phi, trans_data, next_states
#
#     def set_state_min_and_diff(self):
#         self.state_min = np.array([-2.4, self.min_cart_velo, -12*math.pi/180, self.min_pole_velo])
#         self.state_max_min_diff = np.array([4.8, self.max_cart_velo-self.min_cart_velo, 12*2*math.pi/180, self.max_pole_velo-self.min_pole_velo])
#
#     def generate_data_with_one_pass(self):
#         num_data_collected = 0
#         num_states_visited = 0
#         A, b, C = MountainCar.init_abc(self.nFeatures)
#         Phi = []
#         next_states = []
#         trans_data = np.zeros((self.num_data, 4))
#         s0 = self.env.reset()
#         while num_data_collected < self.num_data:
#             action = self.env.action_space.sample()
#             s1, reward, done, _ = self.env.step(action)
#             phi_s0 = self.get_feature(s0, action)
#             phi_s1 = self.get_feature(s1, self.env.action_space.sample())
#
#             A += np.outer(phi_s0, phi_s0 - self.gamma * phi_s1)
#             C += np.outer(phi_s0, phi_s0)
#             b += reward * phi_s0
#
#             next_states.append(s1)
#             trans_data[num_data_collected] = np.array([num_states_visited, 0, reward, num_states_visited + 1])
#             Phi.append(phi_s0)
#             Phi.append(phi_s1)
#             num_data_collected += 1
#             num_states_visited += 2
#             if done:
#                 s0 = self.env.reset()
#         self.env.close()
#         return A, b, C, Phi, trans_data, next_states
#
#     def get_feature(self, state, action):
#         if self.feature_type == 'raw':
#             return self.get_raw_feature(state, action)
#         elif self.feature_type == 'perf':
#             return self.get_perf_feature(state, action)
#         elif self.feature_type == 'rbf':
#             return self.get_rbf_feature(state, action)
#
#     def get_raw_feature(self, state, action):
#         state[1] = np.tanh(state[1]/10)
#         state[3] = np.tanh(state[3]/10)
#         phi = np.zeros(self.nFeatures)
#         phi[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = np.array(state)
#         return phi
#
#     def get_perf_feature(self, state, action):
#         feature = [1]
#         for i in range(4):
#             j = i
#             while j < 4:
#                 #feature.append(state[i]*state[j]) if i==j else feature.append(state[i]*state[j]*2)
#                 feature.append(state[i]*state[j])
#                 j += 1
#         phi = np.zeros(self.nFeatures)
#         phi[action*self.num_feature_per_action:(action+1)*self.num_feature_per_action] = np.array(feature)
#         return phi
#
#     def get_rbf_feature(self, state, action):
#         state = np.divide(state-self.state_min, self.state_max_min_diff)
#         feature = np.square(state[0]-self.cart_position_means) + np.square(state[1] - self.cart_velo_means) + np.square(state[2] - self.pole_angle_means) + np.square(state[3] - self.pole_velo_means)
#         feature = np.append(np.exp(-feature/0.1).flatten(), 1)
#         rbf_feature = np.zeros(self.nFeatures)
#         rbf_feature[action * self.num_feature_per_action:(action + 1) * self.num_feature_per_action] = feature
#         return rbf_feature
#
#     def set_velo_limites(self, init_state):
#         self.max_cart_velo = init_state[1]
#         self.min_cart_velo = init_state[1]
#         self.max_pole_velo = init_state[3]
#         self.min_pole_velo = init_state[3]
#
#     def update_velo_limits(self, state):
#         cur_cart_velo = state[1]
#         cur_pole_velo = state[3]
#         if cur_cart_velo > self.max_cart_velo:
#             self.max_cart_velo = cur_cart_velo
#         elif cur_cart_velo < self.min_cart_velo:
#             self.min_cart_velo = cur_cart_velo
#         if cur_pole_velo > self.max_pole_velo:
#             self.max_pole_velo = cur_pole_velo
#         elif cur_pole_velo < self.min_pole_velo:
#             self.min_pole_velo = cur_pole_velo


    # def generate_data(self, m=0.5, M=0.5, length=0.6, b=0.1,dt=0.01):
    #     g = 9.81
    #     k = 4. * M - m
    #     A = np.array([[1., dt, 0, 0],
    #                   [dt * 3 * (M + m) / k / length, 1 + dt * 3 * b / k, 0, 0],
    #                   [0., 0, 1, dt],
    #                   [dt * 3 * m * g / k, -4 * b / k, 0, 1]])
    #     B = np.array([0., -3 * dt / length / k, 0, dt * 4 / k]).reshape(4, 1)
    #     Q = np.diag([-100., 0., -1, 0])
    #     R = np.ones((1, 1)) * (-0.1)
    #
    #     start_state = np.array([0.0001, 0, 0, 0])
    #     self.sigma = np.array([0.] * 3 + [0.01])
    #     self.gamma = 0.95
    #
    # def solve_LOR(self,mdp, n_iter=100000, eps=1e-14):
    #     P = np.matrix(np.zeros((mdp.nStates, mdp.nStates)))
    #     R = np.matrix(mdp.R)
    #     b = 0.
    #     theta = np.matrix(np.zeros((mdp.nActions, mdp.nStates)))
    #     A = np.matrix(mdp.AA)
    #     B = np.matrix(mdp.B)
    #     for i in range(n_iter):
    #         theta_n = - mdp.gamma * np.linalg.pinv(R + mdp.gamma * B.T * P *
    #                                            B) * B.T * P * A
    #         P_n, b_n = self.bellman_operator(mdp, P, b, theta,gamma=mdp.gamma)  # Q + theta.T * R * theta + gamma * (A+ B * theta).T * P * (A + B * theta)
    #         if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps and np.linalg.norm(theta - theta_n) < eps:
    #             print("Converged estimating V after " + str(i) + "iterations")
    #             break
    #         P = P_n
    #         b = b_n
    #         theta = theta_n
    #     return np.asarray(theta), P, b
    #
    # def bellman_operator(mdp, P, b, theta, gamma, noise=0.):
    #     Q = np.matrix(mdp.Q)
    #     R = np.matrix(mdp.R)
    #     A = np.matrix(mdp.AA)
    #     B = np.matrix(mdp.B)
    #     Sigma = np.matrix(np.diag(mdp.Sigma))
    #     theta = np.matrix(theta)
    #     if noise == 0.:
    #         noise = np.zeros((theta.shape[0]))
    #     S = A + B * theta
    #     C = Q + theta.T * R * theta
    #
    #     Pn = C + gamma * (S.T * np.matrix(P) * S)
    #     bn = gamma * (b + np.trace(np.matrix(P) * np.matrix(Sigma))) \
    #          + np.trace(
    #         (R + gamma * B.T * np.matrix(P) * B) * np.matrix(np.diag(noise)))
    #     return Pn, bn

# #Boyan chain env
# def generate_Boyan_chain(num_samples_to_generate, gamma, save):
#     Phi = [np.array([0, 0, 0, 0]), np.array([0, 0, 1/4, 3/4]), np.array([0, 0, 1/2, 1/2]),
#            np.array([0, 0, 3/4, 1/4]), np.array([0, 0, 1, 0]), np.array([0, 1/4, 3/4, 0]),
#            np.array([0, 1/2, 1/2, 0]), np.array([0, 3/4, 1/4, 0]), np.array([0, 1, 0, 0]),
#            np.array([1/4, 3/4, 0, 0]), np.array([1/2, 1/2, 0, 0]), np.array([3/4, 1/4, 0, 0]),
#            np.array([1, 0, 0, 0])]
#
#     num_states = len(Phi)
#     num_features = len(Phi[0])
#
#     A = np.zeros((num_features, num_features))
#     b = np.zeros(num_features)
#     C = np.zeros((num_features, num_features))
#     Trans_data = np.zeros((num_samples_to_generate,4))
#
#     prev_state = num_states-1
#     for i in range(num_samples_to_generate):
#         action = (1 if prev_state == 1 else np.random.randint(1,high=3))
#         cur_state = prev_state-action
#         reward = (-2 if cur_state == 0 else -3)
#
#         phi_s_prev = Phi[prev_state]
#         phi_s_cur = Phi[cur_state]
#
#         A = A + np.outer(phi_s_prev, phi_s_prev-gamma*phi_s_cur)
#         b = b + reward*phi_s_prev
#         C = C + np.outer(phi_s_prev, phi_s_prev)
#         Trans_data[i, :] = np.array([prev_state, action, reward, cur_state])
#
#         prev_state = (num_states-1 if cur_state==0 else cur_state)
#
#     A = A/num_samples_to_generate
#     b = b/num_samples_to_generate
#     C = C/num_samples_to_generate
#
#     if save:
#         np.save('./MDP_cache/boyan_chain/A', A)
#         np.save('./MDP_cache/boyan_chain/b', b)
#         np.save('./MDP_cache/boyan_chain/C', C)
#         np.save('./MDP_cache/boyan_chain/Trans_data', Trans_data)
#         np.save('./MDP_cache/boyan_chain/Phi', Phi)
#     return A, b, C, Trans_data, Phi