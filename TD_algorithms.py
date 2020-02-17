from optim_alg import *
from torch.utils import data
import progressbar

def initialize_algorithm(self):
    self.theta = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
    self.omega = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
    self.check_pt = self.num_data if self.num_checks == 0 else int(self.num_epoch / self.num_checks)
    self.record_points_before_one_pass = [0]
    self.mspbe_history = torch.unsqueeze(mspbe.calc_mspbe_torch(self, self.rho), 0)
    self.progress_bar = progressbar.ProgressBar(max_value=self.num_epoch + 10)
    self.data_generator = data.DataLoader(dataset_for_td(self), batch_size=self.batch_size, shuffle=True)
    self.num_grad_eval = 0
    self.num_pass = 0
    self.cur_epoch = 0
    self.z = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)

class classical_td(stoc_var_reduce_alg):

    def __init__(self, et_lambda=0, **kwargs):
        stoc_var_reduce_alg.__init__(self, **kwargs)
        self.name = 'classical_td'
        self.td_lambda = et_lambda

    def run(self):
        return stoc_var_reduce_alg.run(self)

    def _run(self):
        stoc_var_reduce_alg.load_mdp_data(self)
        initialize_algorithm(self)

        while self.num_pass < self.num_epoch:
            for batch_r, batch_phi_s_t, batch_phi_s_t_1 in self.data_generator:
                num_data_in_batch = batch_r.shape[0]
                for j in range(num_data_in_batch):
                    r_t = batch_r[j]
                    phi_s_t = batch_phi_s_t[j]
                    phi_s_t_1 = batch_phi_s_t_1[j]
                    self.z = phi_s_t + self.td_lambda*self.gamma*self.z
                    delta_t = r_t + torch.dot(self.gamma*phi_s_t_1-phi_s_t, self.theta)
                    self.theta = self.theta + self.sigma_theta*delta_t * self.z
                self.num_grad_eval += num_data_in_batch
                if self.record_before_one_pass: self.record_value_before_one_pass()
                self.check_complete_data_pass()

                # Temporary
                mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
                print('td mspbe = ' + "{0:.3e}".format(mspbe_at_epoch))
                self.progress_bar.update(self.num_pass)

        stoc_var_reduce_alg.end_of_exp(self)
        return {'theta':self.theta, 'et_lambda':self.td_lambda, 'result':self.result, 'sigma_theta':self.sigma_theta, 'sigma_omega':self.sigma_omega,
                'name': self.name, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac, 'record_points_before_one_pass':self.record_points_before_one_pass}

class tdc(stoc_var_reduce_alg):

    def __init__(self, et_lambda=0, **kwargs):
        stoc_var_reduce_alg.__init__(self, **kwargs)
        self.name = 'tdc'
        self.tdc_lambda = et_lambda

    def run(self):
        return stoc_var_reduce_alg.run(self)

    def _run(self):
        stoc_var_reduce_alg.load_mdp_data(self)
        initialize_algorithm(self)

        while self.num_pass < self.num_epoch:
            for batch_r, batch_phi_s_t, batch_phi_s_t_1 in self.data_generator:
                num_data_in_batch = batch_r.shape[0]
                for j in range(num_data_in_batch):
                    r_t = batch_r[j]
                    phi_s_t = batch_phi_s_t[j]
                    phi_s_t_1 = batch_phi_s_t_1[j]
                    theta_old = self.theta.clone()
                    omega_old = self.omega.clone()
                    delta_t = r_t + torch.dot(self.gamma * phi_s_t_1 - phi_s_t, theta_old)

                    self.z = phi_s_t + self.tdc_lambda * self.gamma * self.z
                    self.theta = theta_old + self.sigma_theta * (delta_t * self.z - self.gamma*(1-self.tdc_lambda) * torch.dot(self.z, omega_old) * phi_s_t_1)
                    self.omega = omega_old + self.sigma_omega * (delta_t * self.z - torch.dot(phi_s_t, omega_old) * phi_s_t)

                    # self.omega = omega_old + self.sigma_omega * (delta_t - torch.dot(phi_s_t, omega_old)) * phi_s_t
                    # self.theta = theta_old + self.sigma_theta * (delta_t * phi_s_t - self.gamma*torch.dot(phi_s_t, omega_old) * phi_s_t_1)

                self.num_grad_eval += num_data_in_batch
                if self.record_before_one_pass: self.record_value_before_one_pass()
                self.check_complete_data_pass()

                # Temporary
                mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
                print('tdc mspbe = ' + "{0:.3e}".format(mspbe_at_epoch) + 'sigma_theta' + "{0:.3e}".format(self.sigma_theta) + 'sigma_omega' + "{0:.3e}".format(self.sigma_omega))
                self.progress_bar.update(self.num_pass)

        stoc_var_reduce_alg.end_of_exp(self)
        return {'theta':self.theta,'omega':self.omega, 'et_lambda':self.tdc_lambda, 'result':self.result, 'sigma_theta':self.sigma_theta, 'sigma_omega':self.sigma_omega,
                'name': self.name, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac, 'record_points_before_one_pass':self.record_points_before_one_pass}

class etd(stoc_var_reduce_alg):

    def __init__(self, et_lambda=0, **kwargs):
        stoc_var_reduce_alg.__init__(self, **kwargs)
        self.name = 'etd'
        self.etd_lambda = et_lambda

    def run(self):
        return stoc_var_reduce_alg.run(self)

    def _run(self):
        stoc_var_reduce_alg.load_mdp_data(self)
        initialize_algorithm(self)
        e = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
        F = 0

        while self.num_pass < self.num_epoch:
            for batch_r, batch_phi_s_t, batch_phi_s_t_1 in self.data_generator:
                num_data_in_batch = batch_r.shape[0]
                for j in range(num_data_in_batch):
                    r_t = batch_r[j]
                    phi_s_t = batch_phi_s_t[j]
                    phi_s_t_1 = batch_phi_s_t_1[j]
                    delta_t = r_t + torch.dot(self.gamma * phi_s_t_1 - phi_s_t, self.theta)

                    F = 1 + self.gamma*F
                    M = self.etd_lambda + (1-self.etd_lambda)*F
                    e = self.gamma*self.etd_lambda*e + M*phi_s_t
                    self.theta = self.theta + self.sigma_theta * delta_t * e

                self.num_grad_eval += num_data_in_batch
                if self.record_before_one_pass: self.record_value_before_one_pass()
                self.check_complete_data_pass()

                # Temporary
                mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
                print('etd mspbe = ' + "{0:.3e}".format(mspbe_at_epoch))
                self.progress_bar.update(self.num_pass)

        stoc_var_reduce_alg.end_of_exp(self)
        return {'theta':self.theta,'omega':self.omega, 'et_lambda':self.etd_lambda, 'result':self.result,
                'sigma_theta':self.sigma_theta, 'sigma_omega':self.sigma_omega, 'record_points_before_one_pass':self.record_points_before_one_pass,
                'name': self.name, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}

class dataset_for_td(data.Dataset):
    def __init__(self, exp):
        self.exp = exp

    def __len__(self):
        return self.exp.num_data

    def __getitem__(self, index):
        trans_data = self.exp.trans_data[index, :]
        s_t = int(trans_data[0])
        s_t_1 = int(trans_data[3])
        r_t = trans_data[2]
        phi_s_t = self.exp.Phi[s_t, :]
        phi_s_t_1 = self.exp.Phi[s_t_1, :]
        return r_t, phi_s_t, phi_s_t_1