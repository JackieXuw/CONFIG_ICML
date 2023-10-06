import numpy as np
import math
import os
import GPy
import safeopt
import scipy as sp
import datetime as dt


cost_funcs = {
        'square': lambda x: np.square(x),
        'exp': lambda x: np.exp(x) - 1,
        'linear': lambda x: x
    }
cost_funcs_inv = {
        'square': lambda x: np.sqrt(x),
        'exp': lambda x: np.log(x+1),
        'linear': lambda x: x
    }


"""
Define some utility functions for the test of safe Bayesian optimization,
constrained Bayesian optimization, and our method.
"""

def try_sample_gp_func(random_sample, num_knots, problem_dim, bounds, config,
                       kernel, numerical_epsilon, noise_var):
    if random_sample:
        sample_list = []
        for i in range(num_knots):
            loc = np.random.uniform(0.0, 1.0, problem_dim)
            x = np.array([bounds[i][0] +
                         (bounds[i][1] - bounds[i][0]) * loc[i]
                          for i in range(problem_dim)]
                         )
            sample_list.append(x)
        knot_x = np.array(sample_list)
    else:
        knot_x = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

    config['knot_x'] = knot_x
    knot_cov = kernel.K(config['knot_x']) + \
        np.eye(knot_x.shape[0]) * numerical_epsilon
    knot_cov_cho = sp.linalg.cho_factor(knot_cov)
    fun = safeopt.sample_gp_function(kernel,
                                     config['bounds'],
                                     noise_var,
                                     config['discretize_num_list']
                                     )
    knot_y = fun(knot_x)
    alpha = sp.linalg.cho_solve(knot_cov_cho, knot_y)

    def obj_f(x):
        x = np.atleast_2d(x)
        y = kernel.K(x, knot_x).dot(alpha)
        y = np.squeeze(y)
        return y

    #config['knot_cov'] = knot_cov
    #config['knot_cov_cho'] = knot_cov_cho
    #config['knot_y'] = knot_y
    #config['knot_min'] = np.min(knot_y)
    #config['alpha'] = alpha
    func_norm = (knot_y.T @ alpha)[0, 0]
    return obj_f, func_norm


def get_config(problem_name, problem_dim=None, gp_kernel=None, init_points_id=0):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    config['problem_name'] = problem_name

    if problem_name == 'sinusodal':
        config['var_dim'] = 2
        config['discretize_num_list'] = [100 for _ in range(config['var_dim'])]
        config['num_constrs'] = 1
        config['bounds'] = [(0, 6), (0, 6)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [20 for _ in range(config['var_dim'])]
        )

        def f(x_f):
            return np.cos(2 * x_f[:, 0]) * np.cos(x_f[:, 1]) + np.sin(x_f[:, 0])

        def g_1(x_g1):
            return np.cos(x_g1[:, 0]) * np.cos(x_g1[:, 1]) - \
                   np.sin(x_g1[:, 0]) * np.sin(x_g1[:, 1]) + 0.2

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[math.pi * 0.5, math.pi * 0.5]])

    if problem_name == 'sinusodal_1d':
        ALPHA = 1
        config['var_dim'] = 1
        config['discretize_num_list'] = [100 * ALPHA
                                         for _ in range(config['var_dim'])]
        config['num_constrs'] = 1
        config['bounds'] = [(0, ALPHA * np.pi)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            [(0, ALPHA * np.pi/6)],
            [20 * ALPHA for _ in range(config['var_dim'])]
        )
        f = lambda x: -np.sin(x[:, 0] / ALPHA)
        g_1 = lambda x: 0.0-np.cos(x[:, 0] / ALPHA)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[0.0]]) * ALPHA

    if problem_name == 'GP_sample_single_func':
        if problem_dim is None:
            problem_dim = 1
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        # Measurement noise
        noise_var = 0.00

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                                  lengthscale=1.0, ARD=True)
        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        safe_margin = 0.2
        func = sample_safe_fun(
            kernel, config, noise_var, gp_kernel, safe_margin, num_bound=10)
        func_min = np.min(func(parameter_set))
        config['f_min'] = func_min
        f = lambda x: func(x, noise=False).squeeze(axis=1)
        g_1 = lambda x: func(x, noise=False).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    if problem_name == 'GP_sample_two_funcs':
        if problem_dim is None:
            problem_dim = 1
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        config['eval_simu'] = False
        config['eta_func'] = lambda t: 3/np.sqrt(t * 1.0)
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                                  lengthscale=1.0, ARD=True)
        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))
        # Generate function with safe initial point at x=0
        def sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin,
                            norm_bound, num_knots=3):
            while True:
                fun, fun_norm = try_sample_gp_func(
                    random_sample=True, num_knots=num_knots,
                    problem_dim=config['var_dim'], bounds=config['bounds'],
                    config=config, kernel=gp_kernel, numerical_epsilon=1e-10,
                    noise_var=1e-5)
                if gp_kernel == 'Gaussian':
                    if fun(0, noise=False) < -safe_margin \
                            and fun_norm <= norm_bound:
                        break
                if gp_kernel == 'poly':
                    if fun(0, noise=False) < -safe_margin \
                           and fun_norm <= norm_bound:
                        break
            return fun

        def sample_bounded_norm_fun(
            kernel, config, noise_var, gp_kernel, safe_margin,
            norm_bound, num_knots=3):
            while True:
                fun, fun_norm = try_sample_gp_func(
                    random_sample=True, num_knots=num_knots,
                    problem_dim=config['var_dim'], bounds=config['bounds'],
                    config=config, kernel=gp_kernel, numerical_epsilon=1e-10,
                    noise_var=1e-5)
                if gp_kernel == 'Gaussian':
                    if fun_norm <= norm_bound:
                        break
                if gp_kernel == 'poly':
                    if fun_norm <= norm_bound:
                        break
            return fun


        constr_func = sample_safe_fun(kernel, config, noise_var, gp_kernel,
                                      safe_margin=0.2, norm_bound=1.0)
        obj_func = sample_bounded_norm_fun(kernel, config, noise_var, gp_kernel,
                                      safe_margin=0.2, norm_bound=1.0)
        func_feasible_min = np.min(
            obj_func(parameter_set)[constr_func(parameter_set)<=0])
        config['f_min'] = func_feasible_min
        f = lambda x: obj_func(x, noise=False).squeeze(axis=1)
        g_1 = lambda x: constr_func(x, noise=False).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    if problem_name == 'GP_sample_single_func_2d':
        if problem_dim is None:
            problem_dim = 2
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [30] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-5, 5)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        # Measurement noise
        noise_var = 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2., \
                              lengthscale=1.0, ARD=True)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        func = sample_safe_fun()
        func_min = np.min(func(parameter_set))
        config['f_min'] = func_min
        f = lambda x: func(x).squeeze(axis=1)
        g_1 = lambda x: func(x).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    return config

if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
