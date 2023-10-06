import numpy as np
import simple_util
import matplotlib.pyplot as plt
import datetime
from test_ApartTherm_util import get_optimizer, get_init_obj_constrs
from test_ApartTherm_util import get_safe_bo_result, get_constrained_bo_result
from test_ApartTherm_util import get_lcb2_result, get_epbo_result
from test_ApartTherm_util import get_pdbo_result
from lcb2 import EPBO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

optimization_config = {
    'eval_budget': 100
}

safe_bo_config = {
    'noise_level': 0.0,
    'kernel_var': 0.1,
    'train_noise_level': 0.0,
    'problem_name': 'energym_apartment_therm_tune_PH'
}

epbo_penalty_list = [0.2, 1.0, 3.0]
epbo_config = safe_bo_config.copy()
epbo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'penalty': 1.0
        }
    )


#violation_aware_opt.gp_obj.plot()

# compare cost of different methods
EPSILON=1e-4
def plot_results(
    safe_bo_total_cost_list, safe_bo_best_obj_list,
    constrained_bo_total_cost_list, constrained_bo_best_obj_list,
    violation_aware_bo_total_cost_list_set, violation_aware_bo_best_obj_list_set):
    plt.figure()
    plt.plot(np.array(safe_bo_total_cost_list)[:,  0], marker='*')
    plt.plot(np.array(constrained_bo_total_cost_list)[:, 0], marker='o')
    legends_list = ['Safe BO', 'Generic Constrained BO']

    for violation_aware_bo_total_cost_list, budget in violation_aware_bo_total_cost_list_set:
        plt.plot(np.array(violation_aware_bo_total_cost_list)[:, 0], marker='+')
        legends_list.append('Violation Aware BO '+str(budget))
        #print(violation_aware_bo_total_cost_list)

    #plt.plot(np.array([vio_budget]*(optimization_config['eval_budget']+1)), marker='v', color='r')
    plt.xlim((0, optimization_config['eval_budget']+1))
    #plt.ylim((0, 500))
    print(legends_list)
    plt.legend(legends_list)
    plt.xlabel('Step')
    plt.ylabel('Cumulative cost')
    plt.savefig('./fig/cost_inc.png', format='png')
    # compare convergence
    plt.figure()
    plt.plot(safe_bo_best_obj_list, marker='*')
    plt.plot(constrained_bo_best_obj_list, marker='o')

    for violation_aware_bo_best_obj_list, budget in violation_aware_bo_best_obj_list_set:
        plt.plot(violation_aware_bo_best_obj_list, marker='+')

    plt.xlim((0, optimization_config['eval_budget']+1))
    #plt.ylim((0, 500))
    #plt.xlim((0, 20))
    plt.legend(legends_list)
    plt.xlabel('step')
    plt.ylabel('Simple Regret')
    plt.savefig('./fig/best_obj.png', format='png')


#get_vabo_result(problem_config, plot=True, vio_budget=10.0)


# In[ ]:



total_eva_num = 100
vio_budgets_list = [0.0, 10.0, 20.0]

#vabo_
#safe_


def run_one_instance(x):
    print('Start running one instance!')
    global vio_budgets_list
    problem_name = 'energym_apartment_therm_tune_PH'
    problem_config = simple_util.get_config(problem_name)
    #try:
    if True:
        print('Start running lcb2 optimizer!')
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs\
            = [[0]], [[0]], [[0]], [[0]], [[0]] # get_lcb2_result(problem_config)

        print('Stop running lcb2 optimizer!')
        epbo_costs_0, epbo_objs_0, epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[0])

        epbo_costs_1, epbo_objs_1, epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[1])

        epbo_costs_2, epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2\
            = get_epbo_result(problem_config, penalty=epbo_penalty_list[2])

        safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
            = [[0]], [[0]], [[0]], [[0]], [[0]] # get_safe_bo_result(problem_config, plot=False)
        con_costs, con_objs, con_opt, con_obj_traj, con_constrs\
            = [[0]], [[0]], [[0]], [[0]], [[0]] # get_constrained_bo_result(problem_config, plot=False)

        pdbo_costs, pdbo_objs, pdbo_opt, pdbo_obj_traj, pdbo_constrs\
            =  [[0]], [[0]], [[0]], [[0]], [[0]] # get_pdbo_result(problem_config)


        vabo_costs_tmp = []
        vabo_objs_tmp = []
        vabo_obj_traj_tmp = []
        vabo_constrs_tmp = []
        #for budget_id in range(len(vio_budgets_list)):
        #    budget = vio_budgets_list[budget_id]
        #    vabo_costs, vabo_objs, vabo_opt, vabo_obj_traj, vabo_constrs = \
        #        get_vabo_result(problem_config, plot=False, vio_budget=budget)
        #    vabo_costs_tmp.append(vabo_costs)
        #    vabo_objs_tmp.append(vabo_objs)
        #    vabo_obj_traj_tmp.append(vabo_obj_traj)
        #    vabo_constrs_tmp.append(vabo_constrs)
            # print(vabo_costs, budget_id)
    #except Exception as e:
    else:
    #    print(e)
        return None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None, None, None
    return safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, \
        vabo_objs_tmp, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, epbo_obj_traj_0, epbo_constrs_0, \
        epbo_obj_traj_1, epbo_constrs_1, epbo_obj_traj_2, epbo_constrs_2


if __name__ == '__main__':
    problem_config = simple_util.get_config('energym_apartment_therm_tune_PH')
    multi_results = []
    for _ in range(1):
        safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, \
            vabo_objs_tmp, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
            con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
            vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, \
            lcb2_obj_traj, lcb2_constrs, epbo_obj_traj_0, epbo_constrs_0, \
            epbo_obj_traj_1, epbo_constrs_1, epbo_obj_traj_2, epbo_constrs_2 \
        = run_one_instance(0)
        multi_results.append((safe_costs, safe_objs, con_costs, con_objs,
                         vabo_costs_tmp, vabo_objs_tmp, pdbo_costs, pdbo_objs,
                         safe_obj_traj, safe_constrs, con_obj_traj,
                          con_constrs, pdbo_obj_traj, pdbo_constrs,
                          vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, \
                          lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs,
                        epbo_obj_traj_0, epbo_constrs_0, epbo_obj_traj_1,
                              epbo_constrs_1, epbo_obj_traj_2, epbo_constrs_2
))

        safe_cost_lists = []
        safe_simple_regret_lists = []
        safe_regret_lists = []
        safe_constrs_lists = []

        con_bo_cost_lists = []
        con_bo_simple_regret_lists = []
        con_bo_regret_lists = []
        con_bo_constrs_lists = []

        vabo_cost_lists = []
        vabo_simple_regret_lists = []
        vabo_regret_lists = []
        vabo_constrs_lists = []

        vabo_cost_lists_set = [[] for _ in range(len(vio_budgets_list))]
        vabo_simple_regret_lists_set = [[] for i in range(len(vio_budgets_list))]
        vabo_regret_lists_set = [[] for _ in range(len(vio_budgets_list))]
        vabo_constrs_lists_set = [[] for _ in range(len(vio_budgets_list))]

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

        for safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, vabo_objs_tmp,\
            pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, con_obj_traj, \
            con_constrs, pdbo_obj_traj, pdbo_constrs, vabo_obj_traj_tmp, \
            vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, \
            lcb2_constrs, epbo_obj_traj_0, epbo_constrs_0, \
        epbo_obj_traj_1, epbo_constrs_1, epbo_obj_traj_2, epbo_constrs_2\
        in multi_results:
            if safe_costs is not None:
                safe_cost_lists.append(safe_costs)
                safe_simple_regret_lists.append(
                    np.array(safe_objs))
                safe_regret_lists.append(
                     - np.array(safe_obj_traj)
                )
                safe_constrs_lists.append(
                    np.array(safe_constrs)
                )

                con_bo_cost_lists.append(con_costs)
                con_bo_simple_regret_lists.append(
                    np.array(con_objs))
                con_bo_regret_lists.append(
                    np.array(con_obj_traj)
                )
                con_bo_constrs_lists.append(
                    np.array(con_constrs)
                )

                pdbo_cost_lists.append(pdbo_costs)
                pdbo_simple_regret_lists.append(
                    np.array(pdbo_objs)
                )
                pdbo_regret_lists.append(
                    np.array(pdbo_obj_traj)
                )
                pdbo_constrs_lists.append(
                    np.array(pdbo_constrs)
                )

                lcb2_cost_lists.append(lcb2_costs)
                lcb2_simple_regret_lists.append(
                    np.array(lcb2_objs))
                lcb2_regret_lists.append(
                    np.array(lcb2_obj_traj)
                )
                lcb2_constrs_lists.append(
                    np.array(lcb2_constrs)
                )


                epbo_regret_lists_0.append(
                    np.array(epbo_obj_traj_0)
                )
                epbo_constrs_lists_0.append(
                    np.array(epbo_constrs_0)
                )

                epbo_regret_lists_1.append(
                    np.array(epbo_obj_traj_1)
                )
                epbo_constrs_lists_1.append(
                    np.array(epbo_constrs_1)
                )


                epbo_regret_lists_2.append(
                    np.array(epbo_obj_traj_2)
                )
                epbo_constrs_lists_2.append(
                    np.array(epbo_constrs_2)
                )


                for budget_id in range(len(vio_budgets_list)):
                    try:
                        vabo_cost_lists_set[budget_id].append(
                            vabo_costs_tmp[budget_id])
                        vabo_simple_regret_lists_set[budget_id].append(
                            np.array(vabo_objs_tmp[budget_id])
                        )
                        vabo_regret_lists_set[budget_id].append(
                            np.array(vabo_obj_traj_tmp[budget_id])
                        )
                        vabo_constrs_lists_set[budget_id].append(
                            np.array(vabo_constrs_tmp[budget_id])
                        )
                    except Exception as e:
                        print(f'Exception {e}.')
                        continue

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

        vabo_ave_cost_arr_set = []
        vabo_ave_simple_regret_arr_set = []
        vabo_ave_regret_arr_set = []
        #for budget_id in range(len(vio_budgets_list)):
        #    vabo_ave_cost_arr_set.append((np.mean(np.array(vabo_cost_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
        #    vabo_ave_simple_regret_arr_set.append((np.mean(np.array(vabo_simple_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
        #    vabo_ave_regret_arr_set.append((np.mean(np.array(vabo_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))

now_time_str = datetime.datetime.now().strftime(
            "%H_%M_%S-%b_%d_%Y")

np.savez(
    f'test_ApartTherm_result_with_only_epbo_PH_{epbo_penalty_list[0]}_{epbo_penalty_list[1]}_{epbo_penalty_list[2]}_{now_time_str}', safe_ave_cost_arr, safe_ave_simple_regret_arr, con_ave_cost_arr, con_ave_simple_regret_arr, vabo_ave_cost_arr_set,
                vabo_ave_simple_regret_arr_set, safe_cost_lists, safe_simple_regret_lists, con_bo_cost_lists, con_bo_simple_regret_lists, vabo_cost_lists, vabo_simple_regret_lists,
            pdbo_cost_arr, pdbo_simple_regret_arr,
            safe_regret_lists, safe_constrs_lists, con_bo_regret_lists,
            con_bo_constrs_lists, pdbo_regret_lists, pdbo_constrs_lists,
            vabo_regret_lists_set, vabo_constrs_lists_set, lcb2_regret_lists,
            lcb2_constrs_lists, epbo_regret_lists_0, epbo_constrs_0,
                 epbo_regret_lists_1, epbo_constrs_1, epbo_regret_lists_2,
                 epbo_constrs_2)


#safe_ave_cost_arr

def plot_cost_SR_scatter(cost_lists, SR_lists, fig_name=None):
    cost_arr = np.array(cost_lists)
    num_tra, num_eval, num_constr = cost_arr.shape

    SR_arr = np.array(SR_lists)

    for i in range(num_constr):
        plt.figure()
        plt.scatter(cost_arr[:,-1,i], SR_arr[:,-1])
        plt.xlabel('Violation cost '+str(i+1))
        plt.ylabel('Simple regret')
        plt.title(fig_name)

    if fig_name is not None:
        plt.savefig('./fig/'+fig_name, format='png')


#plot_cost_SR_scatter(safe_cost_lists, safe_simple_regret_lists, 'Safe BO cost-simple regret')
#plot_cost_SR_scatter(con_bo_cost_lists, con_bo_simple_regret_lists, 'Constrained BO cost-simple regret')
#plot_cost_SR_scatter(vabo_cost_lists_set[0], vabo_simple_regret_lists_set[0], 'Violation-aware BO cost-simple regret 0.0')
#plot_cost_SR_scatter(vabo_cost_lists_set[1], vabo_simple_regret_lists_set[1], 'Violation-aware BO cost-simple regret 10.0')
#plot_cost_SR_scatter(vabo_cost_lists_set[2], vabo_simple_regret_lists_set[2], 'Violation-aware BO cost-simple regret 20.0')

