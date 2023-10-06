from matplotlib import pyplot as plt
import numpy as np
from pdbo.optimization_problem import OptimizationProblem
import vabo
import pdbo
# from .safe_optimizer import *
# from .constrained_bo import *
# from .violation_aware_bo import *


def get_optimizer(optimizer_config, problem_config):
    """
    get the optimizer for experiments.
    """
    optimizer_type = optimizer_config['optimizer_type']
    problem = OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = vabo.safe_optimizer.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = vabo.constrained_bo.ConstrainedBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'violation_aware_bo':
        opt = vabo.violation_aware_bo.ViolationAwareBO(
            problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'pdbo':
        opt = pdbo.pd_bo.PDBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


def get_bo_result(optimizer_config, problem_config, plot=False):
    opt, best_obj_list, total_cost_list = get_optimizer(
        optimizer_config, problem_config)
    for _ in range(optimizer_config['eval_budget']):
        y_obj, constr_vals = opt.make_step()
        total_cost_list.append(opt.cumu_vio_cost)
        best_obj_list.append(opt.best_obj)

    if plot:
        opt.plot()
        for i in range(opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(total_cost_list)[:, i])
        plt.figure()
        plt.plot(best_obj_list)
    return total_cost_list, best_obj_list
