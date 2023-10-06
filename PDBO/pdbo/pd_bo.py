"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
from .base_optimizer import BaseBO


class PDBO(BaseBO):

    def __init__(self, opt_problem, pd_BO_config):
        # optimization problem and measurement noise
        super().__init__(opt_problem, pd_BO_config)

        # dual lambda init
        if 'init_dual' in pd_BO_config.keys():
            self.dual = pd_BO_config['init_dual']
        else:
            self.dual = np.ones(self.opt_problem.num_constrs)

        # lcb = mean - beta * sigma
        if 'beta_func' in pd_BO_config.keys():
            self.beta_func = pd_BO_config['beta_func']
        else:
            self.beta_func = lambda t: 1

        # dual = [dual + eta * g]^+
        if 'eta_func' in pd_BO_config.keys():
            self.eta_func = pd_BO_config['eta_func']
        else:
            self.eta_func = lambda t: 1

        if 'lcb_coef' in pd_BO_config.keys():
            self.lcb_coef = pd_BO_config['lcb_coef']
        else:
            self.lcb_coef = lambda t: 3

        if 'acq_func_type' in pd_BO_config.keys():
            self.acq_func_type = pd_BO_config['acq_func_type']
        else:
            self.acq_func_type = 'LCB'

        self.INF = 1e10
        self.num_eps = 1e-10   # epsilon for numerical value
        self.eta_0 = pd_BO_config['eta_0']
        self.total_eval_num = pd_BO_config['total_eval_num']

        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)
        self.S = None
        self.t = 0
        self.dual_traj = [self.dual]

    def get_acquisition(self, acq_func_type='LCB'):
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

        # calculate LCB of the objective
        trunc_obj_sd = np.maximum(np.sqrt(obj_var), self.num_eps)
        lcb_coef = self.lcb_coef(self.t)
        lcb = obj_mean - lcb_coef * trunc_obj_sd
        obj_mean_range = np.max(obj_mean) - np.min(obj_mean)
        obj_sd_range = np.max(trunc_obj_sd) - np.min(trunc_obj_sd)
        print(f'Obj mean and sd ranges: {obj_mean_range} and {obj_sd_range}')
        # calculate the LCB of the constraints
        trunc_constr_sd = np.maximum(np.sqrt(constrain_var_arr), self.num_eps)
        lcb_coef = self.lcb_coef(self.t)
        constr_lcb = constrain_mean_arr - lcb_coef * trunc_constr_sd
        constr_mean_range = np.max(constrain_mean_arr, axis=0) - \
            np.min(constrain_mean_arr, axis=0)
        constr_sd_range = np.max(trunc_constr_sd, axis=0) - \
            np.min(trunc_constr_sd, axis=0)
        print(f'Constrs mean and sd ranges: {constr_mean_range} and '+
              f'{constr_sd_range}')
        # objective for primal optimization
        primal_obj = lcb + np.sum(self.dual * constr_lcb, axis=1)

        return primal_obj

    def get_beta(self):
        return min(max(0, self.beta_func(self.curr_eval_budget)), 1.0)

    def optimize(self):
        acq_func_type = self.acq_func_type
        if acq_func_type == 'LCB':
            acq = self.get_acquisition(acq_func_type=acq_func_type)
            next_point_id = np.argmin(acq)
        else:
            raise Exception(f'{acq_func_type} acquistion not supported!')

        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self, update_gp=False, gp_package='gpy'):
        x_next, y_obj, constr_vals, vio_cost = self.step_sample_point()
        vio_cost = np.squeeze(vio_cost)
        self.cumu_vio_cost = self.cumu_vio_cost + vio_cost

        # update dual and t
        self.t = self.t + 1
        self.dual = np.maximum(
            self.dual + self.eta_func(self.t) * constr_vals, 0)
        dual_traj = self.dual_traj
        dual_traj.append(self.dual)
        self.dual_traj = dual_traj

        return y_obj, constr_vals
