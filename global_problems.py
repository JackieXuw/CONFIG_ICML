import numpy as np
import math
import os
import GPy
import safeopt
from util import cost_funcs, cost_funcs_inv
from global_functions import *


"""
Define some utility functions for the test of global optimization methods.
"""

def wrap_functions(old_func):
    new_f = lambda x: [old_func(np.squeeze(x))]
    return new_f


def normalize_functions(old_func, X):
    old_value = [old_func(x) for x in X]
    old_min = np.min(old_value)
    old_max = np.max(old_value)
    old_median = 0.5 * (old_max + old_min)
    new_f = lambda x: old_func(x) - old_median
    return new_f


def get_2d_kernel_params(X, func, kernel='Matern52'):
    Y = np.array([[func(x)] for x in X])
    if kernel == 'Gaussian':
        kernel = GPy.kern.RBF(
            input_dim=2,
            variance=1.0,
            lengthscale=3.0,
            ARD=True
        )
        gp = GPy.models.GPRegression(
            X,
            Y,
            kernel,
            noise_var=0.01
        )
        gp.optimize()
        return gp.kern
    if kernel == 'Matern52':
        kernel = GPy.kern.Matern52(
            input_dim=2,
            variance=1.0,
            lengthscale=3.0,
            ARD=True
        )
        gp = GPy.models.GPRegression(
            X,
            Y,
            kernel,
            noise_var=0.01
        )
        gp.optimize()
        return gp.kern



def get_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0, normalize_func=True):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    config['problem_name'] = problem_name
    config['var_dim'] = 2
    config['discretize_num_list'] = [100 for _ in range(config['var_dim'])]
    config['num_constrs'] = 1
    config['bounds'] = [(-10, 10), (-10, 10)]
    config['grid_X'] = safeopt.linearly_spaced_combinations(
        config['bounds'],
        [40 for _ in range(config['var_dim'])]
    )
    config['train_X'] = safeopt.linearly_spaced_combinations(
        config['bounds'],
        [3 for _ in range(config['var_dim'])]
    )
    config['vio_cost_funcs_list'] = [cost_funcs['square']]
    config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]

    if problem_name == 'P1':
        config['obj'] = Branin
        config['constrs_list'] = [SinQ]

    if problem_name == 'P2':
        config['obj'] = Modified_Branin
        config['constrs_list'] = [SinQ]

    if problem_name == 'P3':
        config['obj'] = Branin
        config['constrs_list'] = [Modified_Branin]

    if problem_name == 'P4':
        config['obj'] = Modified_Branin
        config['constrs_list'] = [Modified_Branin]

    if problem_name == 'P5':
        config['obj'] = Branin
        config['constrs_list'] = [Inverted_Bowl]

    if problem_name == 'P6':
        config['obj'] = Modified_Branin
        config['constrs_list'] = [Inverted_Bowl]

    if problem_name == 'P7':
        config['obj'] = Branin
        config['constrs_list'] = [Bowl]

    if problem_name == 'P8':
        config['obj'] = Modified_Branin
        config['constrs_list'] = [Bowl]

    if problem_name == 'P9':
        config['obj'] = lambda x: np.cos(2*x[0])*np.cos(x[1]) + np.sin(x[0])
        con_func = lambda x: np.cos(x[0]+x[1]) + 0.6
        config['constrs_list'] = [con_func]

    if problem_name == 'P10':
        config['obj'] = lambda x: np.sin(x[0]) + x[1]
        con_func = lambda x: np.sin(x[0]) * np.sin(x[1]) + 0.95
        config['constrs_list'] = [con_func]

    if problem_name == 'P9' or problem_name == 'P10':
        config['bounds'] = [(0, 6.0), (0, 6.0)]
        config['grid_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [40 for _ in range(config['var_dim'])]
        )
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [20 for _ in range(config['var_dim'])]
        )
        normalize_func = False

    if normalize_func:
        config['obj'] = normalize_functions(config['obj'], config['grid_X'])
        for k, constr_g in enumerate(config['constrs_list']):
            config['constrs_list'][k] = normalize_functions(
                constr_g, config['grid_X'])


    train_constr = [config['constrs_list'][0](x) for x in config['train_X']]
    smallest_train_constr_id = np.argmin(train_constr)

    if problem_name == 'P9':
        safe_points = config['train_X'][np.array(train_constr) <= 0-0.2]
        num_safe_points = len(safe_points)
        safe_id = np.random.choice(np.arange(num_safe_points))
        config['init_safe_points'] = np.array(
            [safe_points[safe_id]])
    else:
        config['init_safe_points'] = np.array(
            [config['train_X'][smallest_train_constr_id]])

    kernel_list = []
    kernel_list.append(
        get_2d_kernel_params(config['train_X'], config['obj'])
    )
    for k, constr_g in enumerate(config['constrs_list']):
           kernel_list.append(
               get_2d_kernel_params(config['train_X'], constr_g)
           )

    for kernel in kernel_list:
        print(kernel)
    config['kernel'] = kernel_list

    config['obj'] = wrap_functions(config['obj'])
    for k, constr_g in enumerate(config['constrs_list']):
        config['constrs_list'][k] = wrap_functions(constr_g)

    parameter_set = \
            safeopt.linearly_spaced_combinations(
                config['bounds'],
                config['discretize_num_list'])
    obj_func = config['obj']
    constr_func = config['constrs_list'][0]
    obj_func_vals = np.array([
        obj_func(x) for x in parameter_set
    ])
    constr_func_vals = np.array([
        constr_func(x) for x in parameter_set
    ])
    func_feasible_min = np.min(obj_func_vals[
        constr_func_vals <= 0
    ])
    config['f_min'] = func_feasible_min

    return config


if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
