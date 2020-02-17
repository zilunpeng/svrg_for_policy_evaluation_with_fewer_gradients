import mspbe
import torch
import datetime
import numpy as np
from torch.utils import data
import torch.multiprocessing as mp
from optim_alg import *
import progressbar

class svrg(stoc_var_reduce_alg):

    def __init__(self, **kwargs):
        stoc_var_reduce_alg.__init__(self, **kwargs)

    def init_alg(self):
        stoc_var_reduce_alg.init_alg(self)
        self.inner_loop_epoch = int(self.num_data*self.inner_loop_multiplier)
        sampler = data.RandomSampler(torch.arange(self.num_data), replacement=True, num_samples=self.inner_loop_epoch)
        self.data_generator = data.DataLoader(mdp_dataset(self), batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, drop_last=False)
        self.num_grad_eval = 0
        self.num_pass = 0
        self.cur_epoch = 0

    def run(self):
        return stoc_var_reduce_alg.run(self)

    def check_termination_cond(self):
        if self.terminate_if_less_than_epsilon:
            if (self.record_per_epoch and self.cur_epoch % 100 == 0) or (self.record_per_dataset_pass and self.num_pass % 100 == 0):
                mspbe_val = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
                np.save(os.path.join(self.saving_dir_path, 'theta.npy'), self.theta.cpu().numpy())
                np.save(os.path.join(self.saving_dir_path, 'omega.npy'), self.omega.cpu().numpy())
                if mspbe_val < self.policy_eval_epsilon:
                    print(self.name + ' terminate in ' + str(self.cur_epoch) + ' epochs.')
                    return False
                else:
                    print('epoch ' + str(self.cur_epoch) + '. mspbe = %.5f' % (mspbe_val)) if self.record_per_epoch else print('epoch ' + str(self.num_pass) + '. mspbe = %.5f' % (mspbe_val))
                    return True
            else:
                return True
        elif self.record_per_dataset_pass:
            return self.num_pass < self.num_epoch
        elif self.record_per_epoch:
            return self.cur_epoch < self.num_epoch
        else:
            raise ValueError('invalid option')

    def get_batch_abc(self, full_dataset, batch_indicies, batch_abc_size):
        batch_A = torch.zeros([self.nFeatures,self.nFeatures], dtype=torch.float64, device=self.device)
        batch_b = torch.zeros([self.nFeatures], dtype=torch.float64, device=self.device)
        batch_C = torch.zeros([self.nFeatures, self.nFeatures], dtype=torch.float64, device=self.device)
        sub_dataset = data.Subset(full_dataset, batch_indicies)
        data_generator = data.DataLoader(sub_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        for batch_A_t, batch_b_t, batch_C_t, batch_t_m in data_generator:
            bsize = batch_t_m.size()[0]
            for j in range(bsize):
                A_t, b_t, C_t, _ = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                batch_A.add_(A_t)
                batch_b.add_(b_t)
                batch_C.add_(C_t)
        return torch.div(batch_A, batch_abc_size), torch.div(batch_b, batch_abc_size), torch.div(batch_C, batch_abc_size)

    def end_of_epoch(self):
        if self.record_per_epoch:
            mspbe_val = mspbe.calc_mspbe_torch(self, self.rho)
            if self.cur_epoch % self.check_pt == 0: self.check_values_torch(mspbe_val)
            self.mspbe_history = torch.cat((self.mspbe_history, torch.unsqueeze(mspbe_val, 0)))
            self.cur_epoch += 1

    def get_grad_theta_omega_from_batch_abc(self, theta, omega, dataset, batch_inds, batch_abc_size, rho):
        batch_A, batch_b, batch_C = self.get_batch_abc(dataset, batch_inds, batch_abc_size)
        theta_tilde_grad = mspbe.mspbe_grad_theta(theta, omega, batch_A, rho)
        omega_tilde_grad = mspbe.mspbe_grad_omega(theta, omega, batch_A, batch_b, batch_C, self.rho_omega)
        return theta_tilde_grad, omega_tilde_grad

class svrg_classic(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        self.name = 'svrg'

    def run(self):
        return svrg.run(self)

    #temporary method
    def get_pure_mspbe(self):
        A_theta_minus_b = torch.mv(self.A, self.theta) - self.b
        return (1/2) * torch.dot(A_theta_minus_b, torch.mv(self.C_inv, A_theta_minus_b))

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        if self.terminate_if_less_than_epsilon==False: progress_bar = progressbar.ProgressBar(max_value=self.num_epoch*2)

        while self.check_termination_cond():
            theta_tilde = self.theta.clone()
            omega_tilde = self.omega.clone()
            theta_tilde_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, self.A, rho=self.rho)
            omega_tilde_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, self.A, self.b, self.C, self.rho_omega)
            self.num_grad_eval += self.num_data
            if self.record_per_dataset_pass: self.check_complete_data_pass()

            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
                batch_size = batch_t_m.shape[0]
                for j in range(batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    theta_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, rho=self.rho) + theta_tilde_grad - mspbe.mspbe_grad_theta(theta_tilde, omega_tilde, A_t, rho=self.rho)
                    omega_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega) + omega_tilde_grad - mspbe.mspbe_grad_omega(theta_tilde,omega_tilde,A_t,b_t,C_t, self.rho_omega)
                    self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
                    self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
                self.num_grad_eval += batch_size
                if self.record_per_dataset_pass: self.check_complete_data_pass()

            #Temporary
            mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
            print('svrg mspbe = ' + "{0:.3e}".format(mspbe_at_epoch))

            self.end_of_epoch()
            if self.terminate_if_less_than_epsilon == False: progress_bar.update(self.num_pass) if self.record_per_dataset_pass else progress_bar.update(self.cur_epoch)

        svrg.end_of_exp(self)
        return {'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'inner_loop_multiplier':self.inner_loop_multiplier, 'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}

class pdbg(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        self.name = 'pdbg'

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        progress_bar = progressbar.ProgressBar(max_value=self.num_epoch)
        self.A = self.A + self.rho_ac*torch.eye(self.nFeatures, dtype=torch.float32, device=self.device)
        self.C = self.C + self.rho_ac*torch.eye(self.nFeatures, dtype=torch.float32, device=self.device)

        while self.check_termination_cond():
            theta_grad = - torch.mv(torch.t(self.A), self.omega)
            omega_grad = torch.mv(self.A, self.theta) - self.b + torch.mv(self.C, self.omega)
            self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
            self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
            self.end_of_epoch()
            progress_bar.update(self.cur_epoch)
        svrg.end_of_exp(self)
        return {'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac':self.rho_ac}

class svrg_gs(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        self.name = 'svrg_gs'

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        theta_update_counter = 0
        omega_update_counter = 0

        print('before entering loop ' + str(datetime.datetime.now()))
        for i in range(self.num_epoch):
            theta_tilde = self.theta
            omega_tilde = self.omega
            theta_tilde_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, self.A, rho=0)
            omega_tilde_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, self.A, self.b, self.C, self.rho_omega)
            k = 0
            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
                for j in range(self.batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    theta_grad = mspbe.mspbe_stoc_grad_theta_torch(self.omega,A_t) + theta_tilde_grad - mspbe.mspbe_stoc_grad_theta_torch(omega_tilde, A_t)
                    omega_grad = mspbe.mspbe_stoc_grad_omega_torch(self.theta, self.omega, A_t, b_t,C_t) + omega_tilde_grad - mspbe.mspbe_stoc_grad_omega_torch(theta_tilde, omega_tilde, A_t, b_t, C_t)
                    if torch.gt(torch.dot(theta_grad, theta_grad), torch.dot(omega_grad, omega_grad)):
                        self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
                        theta_update_counter += 1
                    else:
                        self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
                        omega_update_counter += 1
                    k += 1
                    if k == self.inner_loop_epoch:
                        self.mspbe_history.append(float(mspbe.calc_mspbe_torch(self, self.rho)))
                        print('before checking')
                        if i % self.check_pt == 0: self.check_values_torch(self.mspbe_history[i])
                        print('finish epoch ' + str(i))
                        break
        print('after loop ' + str(datetime.datetime.now()))
        svrg.end_of_exp(self)
        return {'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'name': self.name, 'msg': self.msg}

class batch_svrg(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        #self.name = 'batch_svrg'+str(self.batch_svrg_init_ratio)+'&'+str(self.batch_svrg_increment_ratio)
        self.name = 'batch_svrg'

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        outer_loop_batch_size = int(self.num_data * self.batch_svrg_init_ratio)
        full_dataset = mdp_dataset(self)

        if self.terminate_if_less_than_epsilon == False: progress_bar = progressbar.ProgressBar(max_value=self.num_epoch*2)
        while self.check_termination_cond():
            theta_tilde = self.theta.clone()
            omega_tilde = self.omega.clone()
            if outer_loop_batch_size>=self.num_data:
                theta_tilde_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, self.A, rho=self.rho)
                omega_tilde_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, self.A, self.b, self.C, self.rho_omega)
                self.num_grad_eval += self.num_data
            else:
                theta_tilde_grad, omega_tilde_grad = self.get_grad_theta_omega_from_batch_abc(self.theta, self.omega, full_dataset, torch.randperm(self.num_data)[:outer_loop_batch_size], outer_loop_batch_size, rho=self.rho)
                torch.cuda.empty_cache()
                self.num_grad_eval += outer_loop_batch_size
            if self.record_per_dataset_pass: self.check_complete_data_pass()

            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
                batch_size = batch_t_m.shape[0]
                for j in range(batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    theta_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, rho=self.rho) + theta_tilde_grad - mspbe.mspbe_grad_theta(theta_tilde, omega_tilde, A_t, rho=self.rho)
                    omega_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega) + omega_tilde_grad - mspbe.mspbe_grad_omega(theta_tilde,omega_tilde,A_t,b_t,C_t, self.rho_omega)
                    self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
                    self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
                self.num_grad_eval += batch_size
                if self.record_per_dataset_pass: self.check_complete_data_pass()
            if self.record_before_one_pass: self.record_value_before_one_pass()

            # Temporary
            mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
            print('batch svrg mspbe = ' + "{0:.3e}".format(mspbe_at_epoch))

            self.end_of_epoch()
            outer_loop_batch_size = int(outer_loop_batch_size * self.batch_svrg_increment_ratio)
            if self.terminate_if_less_than_epsilon == False: progress_bar.update(self.num_pass) if self.record_per_dataset_pass else progress_bar.update(self.cur_epoch)

        svrg.end_of_exp(self)
        if self.record_before_one_pass:
            return {'record_points_before_one_pass':self.record_points_before_one_pass, 'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'batch_svrg_init_ratio':self.batch_svrg_init_ratio, 'batch_svrg_increment_ratio':self.batch_svrg_increment_ratio, 'rho': self.rho}
        else:
            return {'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'batch_svrg_init_ratio':self.batch_svrg_init_ratio, 'batch_svrg_increment_ratio':self.batch_svrg_increment_ratio, 'rho': self.rho}

class gtd2(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        self.name = 'gtd2'

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)

        if self.record_per_dataset_pass or self.terminate_if_less_than_epsilon:
            self.run_with_outerloop()
        elif self.record_per_epoch:
            self.run_with_epoch()
        else:
            raise ValueError('invalid option')

        svrg.end_of_exp(self)
        if self.record_before_one_pass:
            return {'record_points_before_one_pass':self.record_points_before_one_pass, 'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}
        else:
            return {'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega, 'name': self.name, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}

    def run_with_outerloop(self):
        if self.terminate_if_less_than_epsilon == False: progress_bar = progressbar.ProgressBar(max_value=self.num_epoch)
        while self.check_termination_cond():
            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
                batch_size = batch_t_m.shape[0]
                for j in range(batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    self.theta.sub_(torch.mul(mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, rho=self.rho), self.sigma_theta))
                    self.omega.sub_(torch.mul(mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega), self.sigma_omega))
                self.num_grad_eval += batch_size
                if self.record_before_one_pass: self.record_value_before_one_pass()
            self.check_complete_data_pass() #assume svrg inner loop = n*1

            if self.terminate_if_less_than_epsilon == False: progress_bar.update(self.num_pass)

    def run_with_epoch(self):
        progress_bar = progressbar.ProgressBar(max_value=self.num_epoch)
        sampler = data.RandomSampler(torch.arange(self.num_data), replacement=True, num_samples=self.num_epoch)
        self.data_generator = data.DataLoader(mdp_dataset(self), batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, drop_last=False)
        for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
            batch_size = batch_t_m.shape[0]
            for j in range(batch_size):
                A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                self.theta.sub_(torch.mul(mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, rho=self.rho), self.sigma_theta))
                self.omega.sub_(torch.mul(mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega), self.sigma_omega))
                self.end_of_epoch()
            progress_bar.update(self.cur_epoch)

class saga(svrg):
    def __init__(self, **kwargs):
        svrg.__init__(self, **kwargs)
        self.name = 'saga'

    # assume start from zero vectors
    def create_grad_pool(self):
        g_t_theta = torch.zeros([self.nFeatures, self.num_data], dtype=torch.float64, device=self.device)
        g_t_omega = torch.zeros([self.nFeatures, self.num_data], dtype=torch.float64, device=self.device)
        data_generator = data.DataLoader(mdp_dataset(self), batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)
        for batch_A_t, batch_b_t, batch_C_t, batch_t_m in data_generator:
            batch_size = batch_t_m.shape[0]
            for j in range(batch_size):
                A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                g_t_theta[:, t_m] = mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, self.rho)
                g_t_omega[:, t_m] = mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega)
        B_theta = self.one_over_num_data * torch.sum(g_t_theta, 1)
        B_omega = self.one_over_num_data * torch.sum(g_t_omega, 1)
        return g_t_theta, g_t_omega, B_theta, B_omega

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        g_t_theta, g_t_omega, B_theta, B_omega = self.create_grad_pool()
        print('finish generating grad pool' + str(datetime.datetime.now()))
        self.num_grad_eval += self.num_data
        if self.record_per_dataset_pass: self.check_complete_data_pass()

        if self.terminate_if_less_than_epsilon==False: progress_bar = progressbar.ProgressBar(max_value=self.num_epoch+50)
        while self.check_termination_cond():
            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in self.data_generator:
                batch_size = batch_t_m.shape[0]
                for j in range(batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    h_tm_theta = mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, self.rho)
                    h_tm_omega = mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega)
                    g_tm_theta = g_t_theta[:, t_m]
                    g_tm_omega = g_t_omega[:, t_m]
                    theta_grad = self.sigma_theta * (B_theta + h_tm_theta - g_tm_theta)
                    omega_grad = self.sigma_omega * (B_omega + h_tm_omega - g_tm_omega)
                    self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
                    self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
                    B_theta = B_theta + self.one_over_num_data * (h_tm_theta - g_tm_theta)
                    B_omega = B_omega + self.one_over_num_data * (h_tm_omega - g_tm_omega)
                    g_t_theta[:, t_m] = h_tm_theta
                    g_t_omega[:, t_m] = h_tm_omega
                self.num_grad_eval += batch_size
                self.check_complete_data_pass()
                if self.terminate_if_less_than_epsilon == False: progress_bar.update(self.num_pass) if self.record_per_dataset_pass else progress_bar.update(self.cur_epoch)
            # Temporary
            mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
            print('saga sigma_theta =' + str(self.sigma_theta) + ' sigma_omega = ' + str(self.sigma_omega) + ' saga mspbe = %.5f' % (mspbe_at_epoch))

        svrg.end_of_exp(self)
        return {'theta': self.theta, 'omega': self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,
                'name': self.name, 'record_per_dataset_pass': self.record_per_dataset_pass, 'record_per_epoch': self.record_per_epoch,
                'comp_cost': self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}


class scsg(svrg):
    def __init__(self, scsg_batch_size_ratio=0.1, use_geometric_dist=True, **kwargs):
        svrg.__init__(self, **kwargs)
        self.scsg_batch_size_ratio = scsg_batch_size_ratio
        self.use_geometric_dist = use_geometric_dist
        self.name = 'scsg'

    def run(self):
        return svrg.run(self)

    def _run(self):
        svrg.load_mdp_data(self)
        svrg.init_alg(self)
        full_dataset = mdp_dataset(self)
        scsg_batch_size = int(self.num_data * self.scsg_batch_size_ratio)
        geom_dist_p = 1/(scsg_batch_size+1)
        #rho = 1e-2*mspbe.calc_L_rho(self)

        if self.terminate_if_less_than_epsilon==False: progress_bar = progressbar.ProgressBar(max_value=self.num_epoch+50)
        while self.check_termination_cond():
            theta_tilde = self.theta.clone()
            omega_tilde = self.omega.clone()
            theta_tilde_grad, omega_tilde_grad = self.get_grad_theta_omega_from_batch_abc(self.theta, self.omega, full_dataset, torch.randperm(self.num_data)[:scsg_batch_size], scsg_batch_size, self.rho)

            torch.cuda.empty_cache()
            self.num_grad_eval += scsg_batch_size
            if self.record_per_dataset_pass: self.check_complete_data_pass()

            if self.use_geometric_dist:
                inner_loop_epoch = np.random.geometric(geom_dist_p)
            else:
                inner_loop_epoch = int(self.num_data * self.scsg_batch_size_ratio)
            sampler = data.RandomSampler(torch.arange(self.num_data), replacement=True, num_samples=inner_loop_epoch)
            data_generator = data.DataLoader(full_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, drop_last=False)

            for batch_A_t, batch_b_t, batch_C_t, batch_t_m in data_generator:
                batch_size = batch_t_m.shape[0]
                for j in range(batch_size):
                    A_t, b_t, C_t, t_m = svrg.get_stoc_data(self, batch_A_t, batch_b_t, batch_C_t, batch_t_m, j)
                    theta_grad = mspbe.mspbe_grad_theta(self.theta, self.omega, A_t, rho=self.rho) + theta_tilde_grad - mspbe.mspbe_grad_theta(theta_tilde, omega_tilde, A_t, rho=self.rho)
                    omega_grad = mspbe.mspbe_grad_omega(self.theta, self.omega, A_t, b_t, C_t, self.rho_omega) + omega_tilde_grad - mspbe.mspbe_grad_omega(theta_tilde,omega_tilde,A_t,b_t,C_t, self.rho_omega)
                    self.theta.sub_(torch.mul(theta_grad, self.sigma_theta))
                    self.omega.sub_(torch.mul(omega_grad, self.sigma_omega))
            self.num_grad_eval += inner_loop_epoch
            if self.record_per_dataset_pass: self.check_complete_data_pass()
            if self.record_before_one_pass: self.record_value_before_one_pass()

            # Temporary
            mspbe_at_epoch = float(mspbe.calc_mspbe_torch(self, self.rho).cpu().numpy())
            print('scsg ratio = '+ str(self.scsg_batch_size_ratio) + ' sigma_theta =' + str(self.sigma_theta) + ' sigma_omega = ' + str(self.sigma_omega) + ' scsg mspbe = %.5f' % (mspbe_at_epoch))

            self.end_of_epoch()
            if self.terminate_if_less_than_epsilon==False: progress_bar.update(self.num_pass) if self.record_per_dataset_pass else progress_bar.update(self.cur_epoch)

        svrg.end_of_exp(self)
        #Temporary
        if self.record_before_one_pass:
            return {'record_points_before_one_pass':self.record_points_before_one_pass, 'use_geom_dist':self.use_geometric_dist, 'theta':self.theta, 'omega':self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,'name': self.name, 'scsg_batch_size_ratio':self.scsg_batch_size_ratio, 'record_per_dataset_pass':self.record_per_dataset_pass, 'record_per_epoch':self.record_per_epoch, 'comp_cost':self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}
        else:
            return {'use_geom_dist': self.use_geometric_dist, 'theta': self.theta, 'omega': self.omega, 'result': self.result, 'sigma_theta': self.sigma_theta, 'sigma_omega': self.sigma_omega,
                    'name': self.name, 'scsg_batch_size_ratio': self.scsg_batch_size_ratio, 'record_per_dataset_pass': self.record_per_dataset_pass, 'record_per_epoch': self.record_per_epoch, 'comp_cost': self.num_pass, 'rho': self.rho, 'rho_ac': self.rho_ac}

