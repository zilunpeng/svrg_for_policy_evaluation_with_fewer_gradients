import os
import mspbe
import torch
import shutil
import datetime
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

class Experiment_Setup:
    def __init__(self, num_epoch, exp_settings, saving_dir_path=None, multi_process_exps=False, use_gpu=False, num_processes=2, batch_size=1, num_workers=0):
        self.exp_settings = exp_settings
        self.num_epoch = num_epoch
        self.saving_dir_path = saving_dir_path
        self.multi_process_exps = multi_process_exps
        self.use_gpu = use_gpu
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # if init_random:
        #     np.save(self.saving_dir_path+'init_theta.npy', np.random.rand(self.mdp.nFeatures)*2)
        #     np.save(self.saving_dir_path+'init_omega.npy', np.random.rand(self.mdp.nFeatures)*2)

    def run_exps(self, exp_settings):
        return self.run_exps_parallel(exp_settings) if self.multi_process_exps else self.run_exps_seq(exp_settings)

    def exe_run_by_instance(self, ins):
        return ins.run()

    def run_exps_parallel(self, exp_settings):
        instances = []
        for exp_setting in exp_settings:
            self.record_setup(exp_setting)
            method = exp_setting["method"]
            #exp_setting.pop('method', None)
            instance = self.init_ins_from_setup(method, exp_setting)
            instances.append(instance)

        print('before running parallel ' + str(datetime.datetime.now()))
        temp = []
        with mp.get_context("spawn").Pool(processes=self.num_processes, maxtasksperchild=1) as pool:
        #with mp.Pool(processes=self.num_processes, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(self.exe_run_by_instance, instances):
                temp.append(result)
        pool.close()
        pool.join()
        print('after running parallel ' + str(datetime.datetime.now()))

        # processes = [mp.Process(target=ins.run) for ins in instances]
        # for p in processes:
        #     p.start()
        #     p.join()
        #for p in processes: p.join()
        #print('after join')
        return temp

    def run_exps_seq(self, exp_settings):
        results = []
        for exp_setting in exp_settings:
            self.record_setup(exp_setting)
            method = exp_setting["method"]
            #exp_setting.pop('method', None)
            instance = self.init_ins_from_setup(method, exp_setting)
            results.append(instance.run())
        return [result for result in results if result != None]

    def record_setup(self, exp_setting):
        exp_info = open(os.path.join(self.saving_dir_path,'exp_info.txt'),'a')
        for k,v in exp_setting.items():
            exp_info.write(str(k) + ' >>> ' + str(v) + '\n')
        exp_info.write('\n'*3)
        exp_info.close()

    #custom_setup overrides setup
    def init_ins_from_setup(self, init_method, custom_setup):
        setup = {'num_epoch':self.num_epoch, 'saving_dir_path':self.saving_dir_path, 'num_checks':5, 'use_gpu':self.use_gpu, 'batch_size':self.batch_size, 'num_workers':self.num_workers}
        for key in custom_setup:
            setup[key] = custom_setup[key]
        instance = init_method(**setup)
        return instance

    def get_instances_from_queue(self,queue):
        return [queue.get() for i in range(queue.qsize())]

    def build_settings_from_valid_instances(self, instances):
        return [{"method":ins['method'], "theta_ss":ins['sigma_theta'], "omega_ss":ins['sigma_omega']} for ins in instances]

    def get_valid_instances(self, instances):
        valid_instances = []
        for ins in instances:
            if isinstance(ins, str):
                print(ins)
            else:
                valid_instances.append(ins)
        return valid_instances

    def save_results(self, instances):
        results = []
        for instance in instances:
            #results.append(instance)
            results.append({'result':instance['result'], 'sigma_theta':instance['sigma_theta'], 'sigma_omega':instance['sigma_omega'], 'name':instance['name'], 'record_per_dataset_pass':instance['record_per_dataset_pass']})
        results = pd.DataFrame(results)
        results.to_pickle(os.path.join(self.saving_dir_path,'mspbe_results.pkl'))