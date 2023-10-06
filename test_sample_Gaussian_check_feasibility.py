#!/usr/bin/env python
# coding: utf-8

import numpy as np
import safe_optimizer
import simple_util
import optimization_problem
import constrained_bo
from PDBO import pdbo
import violation_aware_bo
from lcb2 import LCB2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

optimization_config = {
    'eval_budget': 100
}
safe_bo_config = {
    'noise_level':0.0,
    'kernel_var':0.1,
    'train_noise_level': 0.0,
    'problem_name':'GP_sample_two_funcs_epsilon_infeasible'
}

def get_optimizer(optimizer_type, optimizer_config, problem_config):
    problem = optimization_problem.OptimizationProblem(problem_config)
    if optimizer_type == 'lcb2':
        opt = LCB2(problem, optimizer_config)
        best_obj_list = [opt.best_obj]

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list

def get_init_obj_constrs(opt):
    init_obj_val_arr, init_constr_val_arr = \
            opt.get_obj_constr_val(opt.x0_arr)
    init_obj_val_list = [init_obj_val_arr[0, 0]]
    init_constr_val_list = [init_constr_val_arr[0, :]]
    return init_obj_val_list, init_constr_val_list

# test LCB2 on the test function
def get_lcb2_result(problem_config):
    lcb2_config = safe_bo_config.copy()
    lcb2_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        }
    )
    lcb2_opt, lcb2_best_obj_list, lcb2_total_cost_list = get_optimizer(
            'lcb2', lcb2_config, problem_config)
    lcb2_opt_obj_list = lcb2_opt.init_obj_val_list
    lcb2_opt_constr_list = lcb2_opt.init_constr_val_list

    for _ in range(optimization_config['eval_budget']):
        y_obj, constr_vals = lcb2_opt.make_step()
        lcb2_total_cost_list.append(
            lcb2_opt.cumu_vio_cost)
        lcb2_best_obj_list.append(lcb2_opt.best_obj)
        lcb2_opt_obj_list.append(y_obj)
        lcb2_opt_constr_list.append(constr_vals)

    lcb2_feasibility_list = lcb2_opt.aux_feasible_history
    return lcb2_total_cost_list, \
        lcb2_best_obj_list, lcb2_opt, \
        lcb2_opt_obj_list, lcb2_opt_constr_list, \
        lcb2_feasibility_list

EPSILON=1e-4
total_eva_num = 100

def run_one_instance(x):
    global vio_budgets_list
    problem_name = 'GP_sample_two_funcs_epsilon_infeasible'
    problem_config = simple_util.get_config(problem_name)
    try:
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs, \
            lcb2_feasibility_list = get_lcb2_result(problem_config)

    except Exception as e:
        constr_func = problem_config['contrs_list'][0]
        print(f'Constraint function value: {constr_func(0.0)}!')
        print(e)
        return None, None, None, None, None, None, None
    return lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, problem_config, lcb2_feasibility_list


multi_results = []
for _ in range(50):
    lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs, \
        problem_config, lcb2_feasibility_list = run_one_instance(0)
    multi_results.append(
        (lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj,
         lcb2_constrs, problem_config, lcb2_feasibility_list)
                         )

    lcb2_cost_lists = []
    lcb2_simple_regret_lists = []
    lcb2_regret_lists = []
    lcb2_constrs_lists = []
    lcb2_feasibility_lists = []
    for lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, \
        lcb2_constrs, problem_config, lcb2_feasibility_list in multi_results:
        if lcb2_costs is not None:
            lcb2_cost_lists.append(lcb2_costs)
            lcb2_simple_regret_lists.append(
                np.array(lcb2_objs)-problem_config['f_min'])
            lcb2_regret_lists.append(
                np.array(lcb2_obj_traj)-problem_config['f_min']
            )
            lcb2_constrs_lists.append(
                np.array(lcb2_constrs)
            )
            lcb2_feasibility_lists.append(
                lcb2_feasibility_list
            )

    np.savez('test_GP_sample_result_check_feasibility', lcb2_regret_lists,
         lcb2_constrs_lists, lcb2_feasibility_lists)
