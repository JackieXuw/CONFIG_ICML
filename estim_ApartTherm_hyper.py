from ApartmentsThermal import get_ApartTherm_kpis
import numpy as np


def sample_batch_grid_data():
    param_bounds = [(0.01, 0.3), (15, 20), (4, 8)]
    grids_per_dim = 5
    params_grids_list = []
    for dim in range(3):
        _lower, _upper = param_bounds[dim]
        _grids = [_lower + k/(grids_per_dim-1)*(_upper - _lower)
                  for k in range(grids_per_dim)]
        params_grids_list.append(_grids)

    param_list = []
    energy_list = []
    dev_list = []

    for i in range(grids_per_dim):
        for j in range(grids_per_dim):
            for k in range(grids_per_dim):
                param_1 = params_grids_list[0][i]
                param_2 = params_grids_list[1][j]
                param_3 = params_grids_list[2][k]
                energy, dev = get_ApartTherm_kpis(
                    controller='PH', params=(param_1, param_2, param_3)
                )
                param_list.append([param_1, param_2, param_3])
                energy_list.append(energy)
                dev_list.append(dev)
    np.savez('./result/apartment_therm_PH', param_list=param_list,
             energy_list=energy_list, dev_list=dev_list)
    return param_list, energy_list, dev_list


param_list, energy_list, dev_list = sample_batch_grid_data()
