import os
import torch
import mspbe
import warnings
import numpy as np
import pandas as pd
from torch.utils import data
import h5py

class stoc_var_reduce_alg:

    def __init__(self, sigma_theta, sigma_omega, num_epoch, saving_dir_path=None, num_checks=20, use_gpu=True, batch_size=1, record_per_dataset_pass=False, batch_svrg_init_ratio=1, batch_svrg_increment_ratio=1, num_workers=0, grid_search=False, terminate_if_less_than_epsilon=False, policy_eval_epsilon=1e-2, rho_multiplier=0, inner_loop_multiplier=1, record_per_epoch=False, method=None, name=None, rho=0, rho_ac=0, rho_omega=0, record_before_one_pass=False, parsing_feature=False):
        self.num_epoch = num_epoch
        self.check_pt_vals = []
        self.num_checks = num_checks
        self.saving_dir_path = saving_dir_path
        self.sigma_theta = sigma_theta
        self.sigma_omega = sigma_omega
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.record_per_dataset_pass = record_per_dataset_pass
        self.batch_svrg_init_ratio = batch_svrg_init_ratio
        self.batch_svrg_increment_ratio = batch_svrg_increment_ratio
        self.num_workers = num_workers
        self.grid_search = grid_search
        self.terminate_if_less_than_epsilon = terminate_if_less_than_epsilon
        self.policy_eval_epsilon = policy_eval_epsilon
        self.rho_multiplier = rho_multiplier
        self.inner_loop_multiplier = inner_loop_multiplier
        self.record_per_epoch = record_per_epoch
        self.rho = rho
        self.rho_ac = rho_ac
        self.rho_omega = rho_omega
        self.record_before_one_pass = record_before_one_pass
        self.parsing_feature = parsing_feature

    def run(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return self._run()
            except Warning as e:
                return {'result':e.args[-1], 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass}
            except ValueError as error:
                return {'result': str(error), 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass}

    def load_mdp_data(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.use_gpu else torch.device('cpu')
        mdp_info = pd.read_pickle(os.path.join(self.saving_dir_path, 'mdp_info.pkl'))
        for attr in mdp_info:
            setattr(self, attr, mdp_info.at[0,attr])
        self.nFeatures = int(self.nFeatures)
        self.num_data = int(self.num_data)
        self.sample_seq = np.random.randint(low=0, high=self.num_data, size=self.num_epoch)
        if os.path.exists(os.path.join(self.saving_dir_path, 'A.npy')):
            self.A = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'A.npy')), dtype=torch.float64, device=self.device)
            self.b = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'b.npy')), dtype=torch.float64, device=self.device)
            self.C = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'C.npy')), dtype=torch.float64, device=self.device)
            self.C_inv = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'C_inv.npy')), dtype=torch.float64, device=self.device)
            self.Phi = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'Phi.npy')), dtype=torch.float64, device=self.device)
            self.trans_data = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'Trans_data.npy')), dtype=torch.float64, device=self.device)
        elif os.path.exists(os.path.join(self.saving_dir_path, 'data.hdf5')):
            f = h5py.File(os.path.join(self.saving_dir_path, 'data.hdf5'), 'r')
            self.A = torch.as_tensor(f.get('A')[0], dtype=torch.float64, device=self.device)
            self.b = torch.as_tensor(f.get('b')[0], dtype=torch.float64, device=self.device)
            self.C = torch.as_tensor(f.get('C')[0], dtype=torch.float64, device=self.device)
            self.C_inv = torch.as_tensor(f.get('C_inv')[0], dtype=torch.float64, device=self.device)
            self.Phi = torch.as_tensor(f.get('phi')[()], dtype=torch.float64, device=self.device)
            self.trans_data = torch.as_tensor(f.get('trans_data')[()], dtype=torch.float64, device=self.device)

    def init_alg(self):
        if os.path.exists(os.path.join(self.saving_dir_path, 'init_theta.npy')) and os.path.exists(os.path.join(self.saving_dir_path, 'init_omega.npy')):
            self.theta = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'init_theta.npy')), dtype=torch.float64, device=self.device)
            self.omega = torch.as_tensor(np.load(os.path.join(self.saving_dir_path, 'init_omega.npy')), dtype=torch.float64, device=self.device)
        else:
            self.theta = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
            self.omega = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
        self.check_pt = self.num_data if self.num_checks == 0 else int(self.num_epoch / self.num_checks)
        if self.rho_multiplier > 0:
            #self.rho = torch.mul(mspbe.calc_eig_max_AtCinvA(self), self.rho_multiplier)
            #self.rho = torch.tensor(0.01, dtype=torch.float32, device=self.device)
            self.rho = self.rho_multiplier
            print(self.rho)
        if self.record_before_one_pass:
            self.record_points_before_one_pass = [0]
        self.mspbe_history = torch.unsqueeze(mspbe.calc_mspbe_torch(self, self.rho),0)
        self.one_over_num_data = torch.tensor(1 / self.num_data, device=self.device)
        #if self.parsing_feature: self.mdp_env = run_pi_lstd.get_pi_env()

    def record_value_before_one_pass(self):
        mspbe_val = mspbe.calc_mspbe_torch(self, self.rho)
        self.mspbe_history = torch.cat((self.mspbe_history, torch.unsqueeze(mspbe_val, 0)))
        self.record_points_before_one_pass.append(self.num_grad_eval)

    def get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j):
        return batch_A_t[j,:,:], batch_b_t[j,:], batch_C_t[j,:,:], batch_t_m[j]

    def check_complete_data_pass(self):
        if self.num_grad_eval >= self.num_data:
            mspbe_val = mspbe.calc_mspbe_torch(self, self.rho)
            if self.num_pass % self.check_pt == 0: self.check_values_torch(mspbe_val)
            self.mspbe_history = torch.cat((self.mspbe_history,torch.unsqueeze(mspbe_val,0)))
            self.num_pass += 1
            if self.record_before_one_pass: self.record_points_before_one_pass.append(self.num_grad_eval)
            self.num_grad_eval = self.num_grad_eval-self.num_data

    def check_values_torch(self, mspbe_history_i):
        # if torch.isinf(mspbe_history_i): raise ValueError(
        #     'mspbe value at check point is inf.')
        if torch.isnan(self.theta).any() or torch.isnan(self.omega).any(): raise ValueError(
            'theta or omega become nan.')
        # if len(self.check_pt_vals) == self.num_checks and np.all(
        #                 np.diff(np.log(np.array(self.check_pt_vals))) > 0):
        #     raise ValueError(
        #     'mspbe value keeps increasing at check points. ' + 'Reject theta=' + str(
        #         float(self.sigma_theta)) + ' omega=' + str(float(self.sigma_omega)))

    def handle_epoch_result(self ,i, batch_j):
        self.batch_result[batch_j] = mspbe.calc_mspbe_torch(self, self.rho)
        if i % self.check_pt == 0: self.check_values_torch(float(mspbe.calc_mspbe_torch(self, self.rho)))

    def end_of_exp(self):
        print(self.name + ' last mspbe is %.5f'%(float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())))
        if (self.rho+self.rho_omega+self.rho_ac) > 0:
            self.result = self.mspbe_history.cpu().numpy()
        else:
            self.result = np.log10(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy()) if self.grid_search else np.log10(self.mspbe_history.cpu().numpy())
        self.theta = self.theta.cpu().numpy()
        self.omega = self.omega.cpu().numpy()
        self.delete_attrs()

    def delete_attrs(self):
        del self.sample_seq
        if hasattr(self, 'data_generator'): del self.data_generator
        if hasattr(self, 'custom_sampler'): del self.custom_sampler

class mdp_dataset(data.Dataset):
    def __init__(self, exp):
        self.exp = exp

    def __len__(self):
        return self.exp.num_data

    def __getitem__(self, index):
        A_t, b_t, C_t = mspbe.get_stoc_abc_torch(self.exp, index)
        return A_t, b_t, C_t, index

class fixed_array_sampler(data.Sampler):
    def __init__(self, sample_seq):
        self.sample_seq = sample_seq

    def __iter__(self):
        return iter(self.sample_seq)

    def __len__(self):
        return len(self.sample_seq)
