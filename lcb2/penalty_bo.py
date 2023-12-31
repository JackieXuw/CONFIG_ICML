"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
from .base_optimizer import BaseBO
from scipy.stats import norm


class EPBO(BaseBO):

    def __init__(self, opt_problem, epbo_config):
        # optimization problem and measurement noise
        super().__init__(opt_problem, epbo_config)

        # Pr(cost <= beta * budget) >= 1 - \epsilon
        if 'beta_func' in epbo_config.keys():
            self.beta_func = epbo_config['beta_func']
        else:
            self.beta_func = lambda t: 1

        self.penalty = epbo_config['penalty']
        self.num_eps = 1e-10   # epsilon for numerical value
        self.t = 0
        self.total_eval_num = epbo_config['total_eval_num']

    def get_acquisition(self, prob_eps=None):
        #if prob_eps is None:
        #    prob_eps = self.prob_eps
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)
        obj_mean = obj_mean + self.gp_obj_mean
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)
            mean = mean + self.gp_constr_mean_list[i]
            constrain_mean_list.append(np.squeeze(mean))
            constrain_var_list.append(np.squeeze(var))

        constrain_mean_arr = np.array(constrain_mean_list).T
        constrain_var_arr = np.array(constrain_var_list).T

        beta = self.beta_func(self.t)
        obj_lcb = obj_mean - beta * np.sqrt(obj_var)

        constrain_lcb_arr = constrain_mean_arr - \
            beta * np.sqrt(constrain_var_arr)

        lcb_feasible = np.prod(1.0 * (constrain_lcb_arr <= 0.0), axis=1)

        return obj_lcb, lcb_feasible, constrain_lcb_arr

    def optimize(self):
        obj_lcb, lcb_feasible, constrain_lcb_arr = self.get_acquisition()
        pos_constr_lcb_arr = np.maximum(constrain_lcb_arr, 0.0)
        next_point_id = np.argmin(
            obj_lcb + self.penalty * np.sum(pos_constr_lcb_arr, axis=1)
        )
        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self, update_gp=False, gp_package='gpy'):
        x_next, y_obj, constr_vals, _ = self.step_sample_point(
            update_hyperparams=update_gp)
        return y_obj, constr_vals
