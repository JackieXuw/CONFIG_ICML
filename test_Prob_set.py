import numpy as np
import safe_optimizer
import simple_util
import optimization_problem
import constrained_bo
import pdbo
import datetime
import matplotlib.pyplot as plt
import violation_aware_bo
from lcb2 import LCB2
from lcb2 import EPBO

optimization_config = {
    'eval_budget': 100
}
import os
from global_problems import get_config
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# test SafeBO on the test function
def get_optimizer(optimizer_type, optimizer_config, problem_config):
    problem = optimization_problem.OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = safe_optimizer.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = constrained_bo.ConstrainedBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'violation_aware_bo':
        opt = violation_aware_bo.ViolationAwareBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'pdbo':
        opt = pdbo.pd_bo.PDBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'lcb2':
        opt = LCB2(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'epbo':
        opt = EPBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


safe_bo_config = {
    'noise_level':0.0,
    'kernel_var':0.1,
    'train_noise_level': 0.0
   # 'problem_name':'GP_sample_two_funcs'
}

epbo_penalty_list = [10.0, 100.0, 1000.0]
epbo_config = safe_bo_config.copy()
epbo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'penalty': 1.0
        }
    )


def get_init_obj_constrs(opt):
    init_obj_val_arr, init_constr_val_arr = \
            opt.get_obj_constr_val(opt.x0_arr)
    init_obj_val_list = [init_obj_val_arr[0, 0]]
    init_constr_val_list = [init_constr_val_arr[0, :]]
    return init_obj_val_list, init_constr_val_list


def get_safe_bo_result(problem_config, plot=False):
    safe_opt, safe_bo_best_obj_list, safe_bo_total_cost_list = \
        get_optimizer('safe_bo', safe_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(safe_opt)

    safe_opt_obj_list = init_obj_val_list
    safe_opt_constr_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
        y_obj, constr_vals = safe_opt.make_step()
        safe_bo_total_cost_list.append(safe_opt.cumu_vio_cost)
        safe_bo_best_obj_list.append(-safe_opt.best_obj)
        safe_opt_obj_list.append(y_obj)
        safe_opt_constr_list.append(constr_vals)
    if plot:
        safe_opt.plot()
        for i in range(safe_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(safe_bo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(safe_bo_best_obj_list)
    return safe_bo_total_cost_list, safe_bo_best_obj_list, safe_opt, \
        safe_opt_obj_list, safe_opt_constr_list

# In[32]:

# test ConstrainedBO on the test function

def get_constrained_bo_result(problem_config, plot=False):
    constrained_opt, constrained_bo_best_obj_list, \
        constrained_bo_total_cost_list = get_optimizer(
            'constrained_bo', safe_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        constrained_opt)
    constrained_opt_obj_list = init_obj_val_list
    constrained_opt_constr_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        y_obj, constr_vals = constrained_opt.make_step()
        constrained_bo_total_cost_list.append(constrained_opt.cumu_vio_cost)
        constrained_bo_best_obj_list.append(constrained_opt.best_obj)
        constrained_opt_obj_list.append(y_obj)
        constrained_opt_constr_list.append(constr_vals)
    if plot:
        constrained_opt.plot()
        for i in range(constrained_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(constrained_bo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(constrained_bo_best_obj_list)
    return constrained_bo_total_cost_list, constrained_bo_best_obj_list, \
        constrained_opt, constrained_opt_obj_list, constrained_opt_constr_list

# test EPBO on the test function
def get_epbo_result(problem_config, plot=False, penalty=1.0):
    epbo_config['penalty'] = penalty
    epbo_opt, epbo_best_obj_list, epbo_total_cost_list = get_optimizer(
            'epbo', epbo_config, problem_config)
    epbo_opt_obj_list = epbo_opt.init_obj_val_list
    epbo_opt_constr_list = epbo_opt.init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        #if True:
        y_obj, constr_vals = epbo_opt.make_step()
        epbo_total_cost_list.append(
            epbo_opt.cumu_vio_cost)
        epbo_best_obj_list.append(epbo_opt.best_obj)
        epbo_opt_obj_list.append(y_obj)
        epbo_opt_constr_list.append(constr_vals)
    if plot:
        epbo_opt.plot()
        for i in range(epbo_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(epbo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(epbo_best_obj_list)
    return epbo_total_cost_list, \
        epbo_best_obj_list, epbo_opt, \
        epbo_opt_obj_list, epbo_opt_constr_list

# test LCB2 on the test function
def get_lcb2_result(problem_config, plot=False):
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
    #if True:
        #if True:
        y_obj, constr_vals = lcb2_opt.make_step()
        lcb2_total_cost_list.append(
            lcb2_opt.cumu_vio_cost)
        lcb2_best_obj_list.append(lcb2_opt.best_obj)
        lcb2_opt_obj_list.append(y_obj)
        lcb2_opt_constr_list.append(constr_vals)
    if plot:
        lcb2_opt.plot()
        for i in range(lcb2_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(lcb2_total_cost_list)[:, i])
        plt.figure()
        plt.plot(lcb2_best_obj_list)
    return lcb2_total_cost_list, \
        lcb2_best_obj_list, lcb2_opt, \
        lcb2_opt_obj_list, lcb2_opt_constr_list

def get_pdbo_result(problem_config, plot=False):
    pdbo_config = safe_bo_config.copy()
    pdbo_config.update({
        'beta_0': 3,
        'eta_0': 2,
        'total_eval_num': optimization_config['eval_budget'],
        'normalize_input': False
        })
    pdbo_opt, pdbo_best_obj_list, \
        pdbo_total_cost_list = get_optimizer(
            'pdbo', pdbo_config, problem_config)
    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        pdbo_opt)

    pdbo_obj_list = init_obj_val_list
    pdbo_constr_val_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        #if True:
        y_obj, constr_vals = pdbo_opt.make_step()
        pdbo_total_cost_list.append(
            pdbo_opt.cumu_vio_cost)
        pdbo_best_obj_list.append(pdbo_opt.best_obj)
        pdbo_obj_list.append(y_obj)
        pdbo_constr_val_list.append(constr_vals)
    if plot:
        pdbo_opt.plot()
        for i in range(pdbo_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(pdbo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(pdbo_best_obj_list)
    return pdbo_total_cost_list, pdbo_best_obj_list, pdbo_opt, \
        pdbo_obj_list, pdbo_constr_val_list

# In[34]:

#violation_aware_opt.gp_obj.plot()


# In[35]:


# compare cost of different methods
EPSILON=1e-4

total_eva_num = 100
vio_budgets_list = [0.0, 10.0, 20.0]

def run_one_instance(x):
    global vio_budgets_list
    problem_name = 'P'+str(x+1)
    problem_config = get_config(problem_name)

    #try:
    if True:
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs\
            = get_lcb2_result(problem_config)

        epbo_costs_0, epbo_objs_0, epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[0])

        epbo_costs_1, epbo_objs_1, epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[1])

        epbo_costs_2, epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[2])

        safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
            = get_safe_bo_result(problem_config, plot=False)
        con_costs, con_objs, con_opt, con_obj_traj, con_constrs\
            = get_constrained_bo_result(problem_config, plot=False)
        pdbo_costs, pdbo_objs, pdbo_opt, pdbo_obj_traj, pdbo_constrs\
            = get_pdbo_result(problem_config)
    else:
        #print(e)
        return None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None, None, None, \
            None, None, None, None, None
    return safe_costs, safe_objs, con_costs, con_objs, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, problem_config, epbo_costs_0, epbo_objs_0, \
        epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, epbo_objs_2, \
        epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2

multi_results = []
num_runs = 30
for run_id in [8] * num_runs:
    safe_costs, safe_objs, con_costs, con_objs, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, problem_config, epbo_costs_0, epbo_objs_0, \
        epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, epbo_objs_2, \
        epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2 = run_one_instance(run_id)
    multi_results.append((safe_costs, safe_objs, con_costs, con_objs, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, problem_config, epbo_costs_0, epbo_objs_0, \
        epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, epbo_objs_2, \
        epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2))

    safe_cost_lists = []
    safe_simple_regret_lists = []
    safe_regret_lists = []
    safe_constrs_lists = []

    con_bo_cost_lists = []
    con_bo_simple_regret_lists = []
    con_bo_regret_lists = []
    con_bo_constrs_lists = []

    pdbo_cost_lists = []
    pdbo_simple_regret_lists = []
    pdbo_regret_lists = []
    pdbo_constrs_lists = []

    lcb2_cost_lists = []
    lcb2_simple_regret_lists = []
    lcb2_regret_lists = []
    lcb2_constrs_lists = []

    epbo_cost_lists_0 = []
    epbo_simple_regret_lists_0 = []
    epbo_regret_lists_0 = []
    epbo_constrs_lists_0 = []

    epbo_cost_lists_1 = []
    epbo_simple_regret_lists_1 = []
    epbo_regret_lists_1 = []
    epbo_constrs_lists_1 = []

    epbo_cost_lists_2 = []
    epbo_simple_regret_lists_2 = []
    epbo_regret_lists_2 = []
    epbo_constrs_lists_2 = []

    for safe_costs, safe_objs, con_costs, con_objs, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, problem_config, epbo_costs_0, epbo_objs_0, \
        epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, epbo_objs_2, \
        epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2 in multi_results:
        if safe_costs is not None:
            safe_cost_lists.append(safe_costs)
            safe_simple_regret_lists.append(
                np.array(safe_objs)-problem_config['f_min'])
            safe_regret_lists.append(
                -problem_config['f_min'] - np.array(safe_obj_traj)
            )
            safe_constrs_lists.append(
                np.array(safe_constrs)
            )

            con_bo_cost_lists.append(con_costs)
            con_bo_simple_regret_lists.append(
                np.array(con_objs)-problem_config['f_min'])
            con_bo_regret_lists.append(
                np.array(con_obj_traj)-problem_config['f_min']
            )
            con_bo_constrs_lists.append(
                np.array(con_constrs)
            )

            pdbo_cost_lists.append(pdbo_costs)
            pdbo_simple_regret_lists.append(
                np.array(pdbo_objs)-problem_config['f_min'])
            pdbo_regret_lists.append(
                np.array(pdbo_obj_traj)-problem_config['f_min']
            )
            pdbo_constrs_lists.append(
                np.array(pdbo_constrs)
            )

            lcb2_cost_lists.append(lcb2_costs)
            lcb2_simple_regret_lists.append(
                np.array(lcb2_objs)-problem_config['f_min'])
            lcb2_regret_lists.append(
                np.array(lcb2_obj_traj)-problem_config['f_min']
            )
            lcb2_constrs_lists.append(
                np.array(lcb2_constrs)
            )

            epbo_cost_lists_0.append(epbo_costs_0)
            epbo_simple_regret_lists_0.append(
                np.array(epbo_objs_0)-problem_config['f_min'])
            epbo_regret_lists_0.append(
                np.array(epbo_obj_traj_0)-problem_config['f_min']
            )
            epbo_constrs_lists_0.append(
                np.array(epbo_constrs_0)
            )

            epbo_cost_lists_1.append(epbo_costs_1)
            epbo_simple_regret_lists_1.append(
                np.array(epbo_objs_1)-problem_config['f_min'])
            epbo_regret_lists_1.append(
                np.array(epbo_obj_traj_1)-problem_config['f_min']
            )
            epbo_constrs_lists_1.append(
                np.array(epbo_constrs_1)
            )

            epbo_cost_lists_2.append(epbo_costs_2)
            epbo_simple_regret_lists_2.append(
                np.array(epbo_objs_2)-problem_config['f_min'])
            epbo_regret_lists_2.append(
                np.array(epbo_obj_traj_2)-problem_config['f_min']
            )
            epbo_constrs_lists_2.append(
                np.array(epbo_constrs_2)
            )

    safe_ave_cost_arr = np.mean(np.array(safe_cost_lists), axis=0)
    safe_ave_simple_regret_arr = np.mean(np.array(safe_simple_regret_lists), axis=0)
    safe_ave_regret_arr = np.mean(np.array(safe_regret_lists), axis=0)

    con_ave_cost_arr = np.mean(np.array(con_bo_cost_lists), axis=0)
    con_ave_simple_regret_arr = np.mean(np.array(con_bo_simple_regret_lists), axis=0)
    con_ave_regret_arr = np.mean(np.array(con_bo_regret_lists), axis=0)

    pdbo_cost_arr = np.mean(np.array(pdbo_cost_lists), axis=0)
    pdbo_simple_regret_arr = np.mean(np.array(pdbo_simple_regret_lists), axis=0)
    pdbo_regret_arr = np.mean(np.array(pdbo_regret_lists), axis=0)

    lcb2_cost_arr = np.mean(np.array(lcb2_cost_lists), axis=0)
    lcb2_simple_regret_arr = np.mean(np.array(lcb2_simple_regret_lists), axis=0)
    lcb2_regret_arr = np.mean(np.array(lcb2_regret_lists), axis=0)

    epbo_cost_arr_0 = np.mean(np.array(epbo_cost_lists_0), axis=0)
    epbo_simple_regret_arr_0 = np.mean(np.array(epbo_simple_regret_lists_0), axis=0)
    epbo_regret_arr_0 = np.mean(np.array(epbo_regret_lists_0), axis=0)

    epbo_cost_arr_1 = np.mean(np.array(epbo_cost_lists_1), axis=0)
    epbo_simple_regret_arr_1 = np.mean(np.array(epbo_simple_regret_lists_1), axis=0)
    epbo_regret_arr_1 = np.mean(np.array(epbo_regret_lists_1), axis=0)

    epbo_cost_arr_2 = np.mean(np.array(epbo_cost_lists_2), axis=0)
    epbo_simple_regret_arr_2 = np.mean(np.array(epbo_simple_regret_lists_2), axis=0)
    epbo_regret_arr_2 = np.mean(np.array(epbo_regret_lists_2), axis=0)

# In[87]:
now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")

np.savez(f'./result/test_Prob_set_result_with_pdbo_{now_time_str}_penaty_{epbo_penalty_list[0]}_{epbo_penalty_list[1]}_{epbo_penalty_list[2]}', safe_ave_cost_arr, safe_ave_simple_regret_arr, con_ave_cost_arr, con_ave_simple_regret_arr,
         safe_cost_lists, safe_simple_regret_lists, con_bo_cost_lists, con_bo_simple_regret_lists,
         pdbo_cost_arr, pdbo_simple_regret_arr,
         safe_regret_lists, safe_constrs_lists, con_bo_regret_lists,
         con_bo_constrs_lists, pdbo_regret_lists, pdbo_constrs_lists,
          lcb2_regret_lists,
         lcb2_constrs_lists, epbo_regret_lists_0, epbo_constrs_lists_0, epbo_regret_lists_1, epbo_constrs_lists_1,
         epbo_regret_lists_2, epbo_constrs_lists_2)

