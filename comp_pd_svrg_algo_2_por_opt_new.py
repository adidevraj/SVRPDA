import numpy as np
import time

def svrpda_1(r_matrix, theta, v, ss_p, ss_d, n, d, m, iters, obj_theta_star, storage_freq, prob_str):

    mean_r_j = ((r_matrix.sum(axis=0)) / n).reshape(d, 1)

    obj_diff_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    oracle_calls_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    computations_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    computations_approx_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    wall_clock_time_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))

    no_oracle_calls = 0
    flag = 0
    computations = 0
    computations_approx = 0
    time_init = time.clock()

    for k in range(1, iters):

        i_k = np.random.randint(0, n)
        j_k = np.random.randint(0, n)

        r_i_k = (r_matrix[i_k, :]).reshape(d, 1)
        r_j_k = (r_matrix[j_k, :]).reshape(d, 1)

        computations_approx += 2 * d

        if flag == 0:
            grad_v_i_k = np.matmul((r_i_k - r_j_k).transpose(), theta)
            v[i_k] = (1 / (1 + 2 * ss_d)) * (v[i_k] + ss_d * grad_v_i_k)
            grad_theta = - r_i_k + (r_i_k - r_j_k)*v[i_k, 0]
            theta = theta - ss_p * grad_theta
        else:
            v_prev_i_k = v[i_k, 0]
            var_red_grad_v_i_k = np.matmul((r_i_k - r_j_k).transpose(), theta - theta_til) + batch_gd_v[i_k]
            v[i_k] = (1 / (1 + 2 * ss_d)) * (v_prev_i_k + ss_d * var_red_grad_v_i_k)
            u_k += (v[i_k] - v_prev_i_k) * (r_i_k - mean_r_j)/n
            var_red_grad_theta = batch_gd_theta_t_1 + u_k
            theta = theta - ss_p * var_red_grad_theta

        if flag == 0:
            no_oracle_calls += 3
            # Dual update:
            computations += 2*d + 6
            # Primal update:
            computations += 4*d
        else:
            no_oracle_calls += 2
            # Accessing the stored value (r_i_k - mean_r_j) counts as zero oracle calls.

            # Dual update:
            computations += 3*d + 6
            # Primal update:
            computations += 7*d + 1

        if np.remainder(k, m) == 0:

            computations_approx += 2 * n * d

            theta_til = np.zeros((d, 1))
            theta_til[:, 0] = theta[:, 0]
            vtil = v

            batch_gd_theta_t_1 = - mean_r_j
            computations += (n-1)*d + 1

            no_oracle_calls += n

            batch_gd_theta_t_2 = np.zeros((d, 1))
            for i in range(n):
                batch_gd_theta_t_2 = batch_gd_theta_t_2 + (1 / n) * ((r_matrix[i, :]).reshape(d, 1) - mean_r_j) * vtil[i]
                computations += 2 * d + 1

            # Total number of additions in the above loop:
            computations += (n - 1) * d

            no_oracle_calls += n
            u_k = batch_gd_theta_t_2

            batch_gd_v = np.zeros((n, 1))
            for i in range(n):
                r_i = (r_matrix[i, :]).reshape(d, 1)
                batch_gd_v[i] = np.matmul((r_i - mean_r_j).transpose(), theta_til)
                computations += d

            no_oracle_calls += n

            flag = 1

        if np.remainder(k, storage_freq) == 0:

            print(k)

            obj_func_t_1 = - np.matmul(mean_r_j.transpose(), theta)

            obj_func_t_2 = 0
            for i in range(n):
                obj_func_t_2 = obj_func_t_2 + (np.matmul((r_matrix[i, :]).reshape(1, d), theta) + obj_func_t_1)**2
            obj_func_t_2 = obj_func_t_2/n

            obj_func_val = obj_func_t_1 + obj_func_t_2

            obj_diff_vec[int(round(k/storage_freq)) - 1, 0] = (obj_func_val - obj_theta_star)
            print(obj_diff_vec[int(round(k/storage_freq)) - 1, 0])
            oracle_calls_vec[int(round(k/storage_freq)) - 1] = no_oracle_calls
            print(oracle_calls_vec[int(round(k/storage_freq)) - 1])
            computations_vec[int(round(k/storage_freq)) - 1] = computations
            computations_approx_vec[int(round(k/storage_freq)) - 1] = computations_approx
            wall_clock_time_vec[int(round(k/storage_freq)) - 1] = time.clock() - time_init

    save_str = './result/comp_pd_svrg_algo_2_por_opt_' + prob_str

    data_saving = np.zeros((len(obj_diff_vec), 5))
    data_saving[:, 0] = obj_diff_vec[:, 0]
    data_saving[:, 1] = oracle_calls_vec[:, 0]
    data_saving[:, 2] = computations_approx_vec[:, 0]
    data_saving[:, 3] = wall_clock_time_vec[:, 0]

    data_saving[:, 4] = computations_vec[:, 0]

    # np.save(save_str, data_saving)

    return theta, obj_diff_vec, oracle_calls_vec

