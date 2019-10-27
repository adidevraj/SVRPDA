import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from bgd_portfolio_opt import bgd_por_opt
from comp_pd_svrg_algo_2_por_opt_new import svrpda_1
from comp_pd_svrg_algo_3_por_opt_new import svrpda_2
from comp_svrg_por_opt_algo_2_lian import comp_svrg_por_opt_algo_2
from comp_svrg_por_opt_algo_3_lian import comp_svrg_por_opt_algo_3
import h5py


def main():

    plot = 1

    # Probname = ['Asia_Pacific_ex_Japan_ME.mat', 'Europe_ME.mat', 'Global_ex_US_ME.mat', 'Global_ME.mat', 'Japan_ME.mat',
    #             'North_America_ME.mat', 'Asia_Pacific_ex_Japan_OP.mat', 'Europe_OP.mat', 'Global_ex_US_OP.mat',
    #             'Global_OP.mat', 'Japan_OP.mat', 'North_America_OP.mat', 'Asia_Pacific_ex_Japan_INV.mat',
    #             'Europe_INV.mat', 'Global_ex_US_INV.mat', 'Global_INV.mat', 'Japan_INV.mat', 'North_America_INV.mat']

    # Choose the data set you want to run the algorithsm on:

    Probname = ['Asia_Pacific_ex_Japan_ME.mat', 'Europe_ME.mat', 'Global_ex_US_ME.mat', 'Global_ME.mat', 'Japan_ME.mat',
                'North_America_ME.mat']

    nprob = len(Probname)
    problist = np.arange(nprob)

    for prob_i in range(problist):

        prob_str = Probname[prob_i]

        print(prob_str)

        f = h5py.File(prob_str, 'r')
        data = f.get('data')
        data = np.array(data)  # For converting to numpy array

        d, n = np.shape(data)

        print(n)

        # Copy the n x d reward matrix:
        r_matrix = data.T

        # Calculating theta^*:

        mean_r_j = ((r_matrix.sum(axis=0)) / n).reshape(d, 1)
        print(mean_r_j)
        outer_prod_matrix_r = np.zeros((d, d))
        for i in range(n):
            outer_prod_matrix_r += np.matmul((r_matrix[i, :]).reshape(d, 1), (r_matrix[i, :]).reshape(1, d))
        emp_cov_matrix = outer_prod_matrix_r / n - np.matmul(mean_r_j.reshape(d, 1), mean_r_j.reshape(1, d))
        theta_star = np.linalg.solve(2 * emp_cov_matrix, mean_r_j)

        print(theta_star)
        print(mean_r_j)

        # Obtain the condition number:

        eigs = np.linalg.eig(emp_cov_matrix)[0]
        print(max(eigs))
        print(min(eigs))
        print('kappa')
        print(max(eigs) / min(eigs))

        # Calculate the function value evaluated at theta^*:

        obj_func_theta_star_t_1 = - np.matmul(mean_r_j.transpose(), theta_star)
        obj_func_theta_star_t_2 = 0
        for i in range(n):
            obj_func_theta_star_t_2 = obj_func_theta_star_t_2 + (np.matmul((r_matrix[i, :]).reshape(1, d), theta_star)
                                                                 + obj_func_theta_star_t_1) ** 2
        obj_func_theta_star_t_2 = obj_func_theta_star_t_2 / n
        obj_func_theta_star = obj_func_theta_star_t_1 + obj_func_theta_star_t_2

        print(obj_func_theta_star)

        # Obtain the objective gap as a function of iterations for gradient descent:

        theta_init = np.random.random((d, 1))
        storage_freq = 100

        s_size_p = 0.01
        iters = 300

        theta_vec_bgd, grad_vec_bgd, objec_error_gd, oracle_call_gd = \
            bgd_por_opt(r_matrix, theta_init, s_size_p, n, d, iters, obj_func_theta_star, prob_str)

        # Next implement our algorithms:

        # This is SVRPDA I:

        msvrg = n  # Number of inner-loops per one outer-loop

        v_init = np.random.random((n, 1))  # Initialize the dual variables

        iters = 500000  # Total number of iterations
        s_size_p = 0.0003  # Primal update step-size
        s_size_d = 100  # Dual update step-size

        theta_vec_compd_2, obj_error_vec_compd_2, oracle_call_comp_pd_2 = \
            svrpda_1(r_matrix, theta_init, v_init, s_size_p, s_size_d, n, d, msvrg, iters,
                                        obj_func_theta_star, storage_freq, prob_str)

        # This is SVRPDA II:

        iters = 1000000  # Total number of iterations
        s_size_p = 0.0003  # Primal update step-size
        s_size_d = 100  # Dual update step-size

        theta_vec_compd_3, obj_error_vec_compd_3, oracle_call_comp_pd_3 = \
            svrpda_2(r_matrix, theta_init, v_init, s_size_p, s_size_d, n, d, msvrg, iters,
                                        obj_func_theta_star, storage_freq, prob_str)

        # This is Comp-SVRG-1 (Algorithm 2 of Lian et al.):

        iters = 400000
        s_size = 0.0003
        A = 6

        theta_vec_comp_svrg_algo_2, obj_error_vec_comp_svrg_algo_2, oracle_calls_vec_comp_svrg_algo_2 = \
            comp_svrg_por_opt_algo_2(r_matrix, theta_init, s_size, n, d, msvrg, iters, storage_freq, obj_func_theta_star
                                     , A, prob_str)

        # This is Comp-SVRG-2 (Algorithm 3 of Lian et al.):

        iters = 400000
        s_size = 0.0004
        A = 6
        B = 6

        theta_vec_comp_svrg, obj_error_vec_comp_svrg, oracle_calls_vec_comp_svrg = \
            comp_svrg_por_opt_algo_3(r_matrix, theta_init, s_size, n, d, msvrg, iters, storage_freq, obj_func_theta_star,
                                     A, B, prob_str)

        if plot == 1:

            plt.semilogy(oracle_calls_vec_comp_svrg_algo_2, obj_error_vec_comp_svrg_algo_2,
                         label='Comp SVRG - Lian - Algo 2')

            plt.semilogy(oracle_calls_vec_comp_svrg, obj_error_vec_comp_svrg,
                         label='Comp SVRG - Lian - Algo 3')

            plt.semilogy(oracle_call_comp_pd_2, obj_error_vec_compd_2,
                         label='SVRPDA I')

            plt.semilogy(oracle_call_comp_pd_3, obj_error_vec_compd_3,
                         label='SVRPDA II')

            plt.semilogy(oracle_call_gd, objec_error_gd,
                         label='Batch GD')

            plt.ylabel('Objective gap')
            plt.xlabel('Number of oracle calls')
            plt.legend(loc='upper right')
            plt.show()


if __name__ == '__main__':
    main()
