import numpy as np
import time


def comp_svrg_por_opt_algo_3(r_matrix, theta, s_size, n, d, m, iters, storage_freq, obj_theta_star, A, B, prob_str):

    print('Algo 3 Lian et all')

    obj_diff_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    oracle_calls_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    computations_approx_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))
    wall_clock_time_vec = np.zeros((int(round(iters/storage_freq)) - 1, 1))

    no_oracle_calls = 0
    computations_approx = 0
    time_init = time.clock()

    mean_r_j = ((r_matrix.sum(axis=0)) / n).reshape(d, 1)

    wtil = np.zeros((d, 1))
    G_til = np.zeros((d+1, 1))
    G_til_prime = np.zeros((d+1, d))
    f_til_prime = np.zeros((d, 1))

    print(np.shape(wtil))

    for k in range(1, iters):

        computations_approx += d * (d+1)

        sampled_indices_A = np.random.randint(n, size=A)
        sampled_indices_B = np.random.randint(n, size=B)

        G_hat_k = np.zeros((d+1, 1))

        for j in sampled_indices_A:

            r_j = (r_matrix[j, :]).reshape(d, 1)

            G_hat_k[0:d, 0] += (theta - wtil)[:, 0]

            G_hat_k[d, 0] += np.matmul((theta - wtil).transpose(), r_j)[0, 0]

            no_oracle_calls += 2

        G_hat_k = G_til + G_hat_k/A

        G_hat_prime_k = np.zeros((d+1, d))

        # for i in sampled_indices_B:
        #
        #     r_i_transpose = (r_matrix[i, :]).reshape(1, d)
        #     G_hat_prime_k[d, :] += (r_i_transpose - r_i_transpose)[0, :]

            # no_oracle_calls += 0

        # G_hat_prime_k[d, :] = G_hat_prime_k[d, :]/B

        # Useless; All elements of the the matrix are equal to zero before adding G_til_prime

        G_hat_prime_k = G_til_prime + G_hat_prime_k

        i_k = np.random.randint(n)
        r_i_k_transpose = (r_matrix[i_k, :]).reshape(1, d)

        y = G_hat_k

        Grad_F_i_k_G_hat_k = np.zeros((d+1, 1))
        Grad_F_i_k_G_hat_k[0:d, 0] = (2 * (np.matmul(r_i_k_transpose, y[0:d, 0]) - y[d, 0]) *
                                      r_i_k_transpose.reshape(d, 1))[:, 0]
        Grad_F_i_k_G_hat_k[d, 0] = -1 - 2*(np.matmul(r_i_k_transpose, y[0:d, 0]) - y[d, 0])

        no_oracle_calls += 1

        y = G_til

        Grad_F_i_k_G_til = np.zeros((d+1, 1))
        Grad_F_i_k_G_til[0:d, 0] = (2 * (np.matmul(r_i_k_transpose, y[0:d, 0]) - y[d, 0]) *
                                    r_i_k_transpose.reshape(d, 1))[:, 0]
        Grad_F_i_k_G_til[d, 0] = -1 - 2*(np.matmul(r_i_k_transpose, y[0:d, 0]) - y[d, 0])

        no_oracle_calls += 1

        f_hat_prime_k = np.matmul(G_hat_prime_k.transpose(), Grad_F_i_k_G_hat_k) - np.matmul(
            G_til_prime.transpose(), Grad_F_i_k_G_til) + f_til_prime

        theta = theta - s_size*f_hat_prime_k

        if np.remainder(k, m) == 0:

            computations_approx += d * (d+1) * n

            wtil = theta
            G_til = np.zeros((d+1, 1))

            G_til[0:d, 0] = wtil[:, 0]
            G_til[d, 0] = np.matmul(wtil.transpose(), mean_r_j)

            no_oracle_calls += n

            G_til_prime = np.zeros((d+1, d))

            G_til_prime[0:d, 0:d] = np.eye(d)
            G_til_prime[d, :] = mean_r_j.transpose()

            no_oracle_calls += n

            Grad_F_G_til = np.zeros((d+1, 1))

            for i in range(n):
                r_i_transpose = (r_matrix[i, :]).reshape(1, d)

                y = G_til

                Grad_F_i_G_til = np.zeros((d+1, 1))
                Grad_F_i_G_til[0:d, 0] = (2 * (np.matmul(r_i_transpose, y[0:d, 0]) - y[d, 0]) *
                                          r_i_transpose.reshape(d, 1))[:, 0]
                Grad_F_i_G_til[d, 0] = -1 - 2 * (np.matmul(r_i_transpose, y[0:d, 0]) - y[d, 0])

                Grad_F_G_til += Grad_F_i_G_til

            Grad_F_G_til = Grad_F_G_til/n

            no_oracle_calls += n

            f_til_prime = np.matmul(G_til_prime.transpose(), Grad_F_G_til)

        if np.remainder(k, storage_freq) == 0:

            obj_func_t_1 = - np.matmul(mean_r_j.transpose(), theta)

            obj_func_t_2 = 0
            for i in range(n):
                obj_func_t_2 = obj_func_t_2 + (np.matmul((r_matrix[i, :]).reshape(1, d), theta) + obj_func_t_1)**2
            obj_func_t_2 = obj_func_t_2/n

            obj_func_val = obj_func_t_1 + obj_func_t_2

            # print(obj_func_val)

            obj_diff_vec[int(round(k/storage_freq)) - 1, 0] = (obj_func_val - obj_theta_star)
            print(k)
            print(obj_diff_vec[int(round(k/storage_freq)) - 1, 0])
            oracle_calls_vec[int(round(k/storage_freq)) - 1] = no_oracle_calls
            print(oracle_calls_vec[int(round(k/storage_freq)) - 1, 0])
            computations_approx_vec[int(round(k/storage_freq)) - 1] = computations_approx
            wall_clock_time_vec[int(round(k/storage_freq)) - 1] = time.clock() - time_init

    save_str = './result/comp_svrg_por_opt_algo_3_lian_' + prob_str

    data_saving = np.zeros((len(obj_diff_vec), 4))
    data_saving[:, 0] = obj_diff_vec[:, 0]
    data_saving[:, 1] = oracle_calls_vec[:, 0]
    data_saving[:, 2] = computations_approx_vec[:, 0]
    data_saving[:, 3] = wall_clock_time_vec[:, 0]

    np.save(save_str, data_saving)

    return theta, obj_diff_vec, oracle_calls_vec
