import time
from math import log
from random import uniform

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

class glmMCMC:
    def __init__(self, design_mat_data, response_data, glmclass, initial_param, rng_seed=2021):
        self.design_mat_data = design_mat_data
        self.response_data = response_data
        self.initial = initial_param
        self.num_dim = len(initial_param)
        
        self.glmclass = glmclass
        self.MC_sample = [initial_param]

        self.num_total_iters = 0
        self.num_accept = 0

        self.random_seed = rng_seed
        self.random_gen = np.random.default_rng(seed=rng_seed)

    def proposal_sampler(self, last_param, cov_rate=0.01):
        cov_mat = np.identity(len(last_param))*cov_rate
        return self.random_gen.multivariate_normal(last_param, cov_mat)

    def proposal_sampler_onlyspecificdim(self, last_param, dim_list, cov_rate=0.01):
        result = [val for val in last_param]
        for i in dim_list:
            result[i] = self.random_gen.normal(last_param[i], cov_rate)
        # print(dim_list, result, last_param) #for debug
        return result

    def log_proposal_pdf(self, from_param, to_param):
        return 0 #symmetric proposal distribution

    def log_prior_pdf(self, param_val):
        return 0 #impropoer uniform

    def log_likelihood(self, param_val):
        if self.glmclass == 'binary':
            each_data = self.response_data * np.matmul(self.design_mat_data, param_val) - np.log(np.exp(np.matmul(self.design_mat_data, param_val))+1)
            return np.sum(each_data)
            
        elif self.glmclass == 'poisson':
            each_data = self.response_data * np.matmul(self.design_mat_data, param_val) - np.exp(np.matmul(self.design_mat_data, param_val))
            #factorial term (-log(factorial(self.response_data))) 필요없는듯. 어차피 캔슬됨
            return np.sum(each_data)

        elif self.glmclass == 'gaussian':
            pass
        else:
            raise ValueError("improper glmclass")


    def log_target_pdf(self, param_val):
        return self.log_likelihood(param_val) + self.log_prior_pdf(param_val)


    def log_r_calculator(self, candid, last):
        log_r = (self.log_target_pdf(candid) - self.log_proposal_pdf(from_param=last, to_param=candid) - \
             self.log_target_pdf(last) + self.log_proposal_pdf(from_param=candid, to_param=last))
        return log_r

    def sampler(self):
        last = self.MC_sample[-1]
        candid = self.proposal_sampler(last) #기존 state 집어넣게
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.MC_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.MC_sample.append(last)
            self.num_total_iters += 1
    
    def specificdim_sampler(self, dim_list, cov_rate):
        last = self.MC_sample[-1]
        candid = self.proposal_sampler_onlyspecificdim(last, dim_list, cov_rate)
        unif_sample = uniform(0, 1)
        log_r = self.log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.MC_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.MC_sample.append(last)
            self.num_total_iters += 1


    
    def generate_samples(self, num_samples, pid=None, verbose=True):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler()

            if i%500 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%500 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        if pid is not None and verbose: #여기 verbose 추가함
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose: #여기 verbose 추가함
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")


    def generate_samples_with_dimgroup(self, num_samples, group_thres, pid=None, verbose=True):
        start_time = time.time()
        for i in range(1, num_samples):
            if i%2 == 0:
                dim_idx_list = [i for i in range(0,group_thres)]
                self.specificdim_sampler(dim_idx_list, cov_rate=0.1)
            else:
                dim_idx_list = [i for i in range(group_thres,self.num_dim)]
                self.specificdim_sampler(dim_idx_list, cov_rate=0.005)

            if i%500 == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%500 == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        if pid is not None and verbose: #여기 verbose 추가함
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose: #여기 verbose 추가함
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")


    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]

    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]
    
    
    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def get_sample_mean(self):
        #burnin자르고 / thining 이후 쓸것
        mean_vec = []
        for i in range(self.num_dim):
            would_cal_mean = self.get_specific_dim_samples(i)
            mean_vec.append(np.mean(would_cal_mean))
        return mean_vec

    def show_traceplot(self):
        grid_column= int(self.num_dim**0.5)
        grid_row = int(self.num_dim/grid_column)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        if grid_column*grid_row < self.num_dim:
            grid_row +=1
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            dim_samples = self.get_specific_dim_samples(i)
            plt.ylabel(str(i)+"-th dim")
            plt.plot(range(len(dim_samples)),dim_samples)
        plt.show()


    def show_hist(self):
        grid_column= int(self.num_dim**0.5)
        grid_row = int(self.num_dim/grid_column)
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        if grid_column*grid_row < self.num_dim:
            grid_row +=1
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            dim_samples = self.get_specific_dim_samples(i)
            plt.ylabel(str(i)+"-th dim")
            plt.hist(dim_samples, bins=100)
        plt.show()

    def get_autocorr(self, dim_idx, maxLag):
        y = self.get_specific_dim_samples(dim_idx)
        acf = []
        y_mean = np.mean(y)
        y = [elem - y_mean  for elem in y]
        n_var = sum([elem**2 for elem in y])
        for k in range(maxLag+1):
            N = len(y)-k
            n_cov_term = 0
            for i in range(N):
                n_cov_term += y[i]*y[i+k]
            acf.append(n_cov_term / n_var)
        return acf

    def show_acf(self, maxLag):
        grid_column= int(self.num_dim**0.5)
        grid_row = int(self.num_dim/grid_column)
        if grid_column*grid_row < self.num_dim:
            grid_row +=1
        subplot_grid = [i for i in range(maxLag+1)]
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i in range(self.num_dim):
            plt.subplot(grid_row, grid_column, i+1)
            acf = self.get_autocorr(i, maxLag)
            plt.ylabel(str(i)+"-th dim")
            plt.ylim([-1,1])
            plt.bar(subplot_grid, acf, width=0.3)
            plt.axhline(0, color="black", linewidth=0.8)
        plt.show()

if __name__=="__main__":
    design_mat = np.array([
        [1,1],
        [1,2],
        [1,3],
        [1,4],
        [1,5],
        [1,6],
        [1,7],
        [1,8],
    ])
    response_mat = np.array([1,1,0,1,0,0,0,0])
    init_param_val = [0,0]
    test_inst = glmMCMC(design_mat, response_mat, 'poisson', init_param_val)
    test_inst.generate_samples(100)
    test_inst.show_traceplot()
    test_inst.show_hist()
    print(test_inst.get_sample_mean())
