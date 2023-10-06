from energym.examples.Controller import LabController
import energym
import pandas as pd
Celcius_to_Kelvin = 273.15


class PHHouseController(object):
    """Rule-based controller for heat pump control.

    Attributes
    ----------
    controls : list of str
        List of control inputs.
    observations : list of str
        List of zone temperature observations
    tol1 : float
        First threshold for deviation from the goal temperature.
    tol2 : float
        Second threshold for deviation from the goal temperature.
    nighttime_setback : bool
        Whether to use a nighttime setback.
    nighttime_start : int
        Hour to start the nighttime setback.
    nighttime_end : int
        Hour to end the nighttime setback.
    nighttime_temp : float
        Goal temperature during nighttime setback

    Methods
    -------
    get_control(obs, temp_sp, hour)
        Computes the control actions.
    """

    def __init__(
        self,
        control_list,
        P,
        set_point,
        nighttime_setback=True,
        nighttime_start=17,
        nighttime_end=6,
        nighttime_temp=18,
    ):
        """
        Parameters
        ----------
        control_list : list of str
            List containing all inputs
        P : float
            Gain for the P-controller.
        set_point : float
            Daytime temperature set point.
        nighttime_setback : bool, optional
            Whether to use a nighttime setback, by default False.
        nighttime_start : int, optional
            Hour to start the nighttime setback, by default 17
        nighttime_end : int, optional
            Hour to end the nighttime setback, by default 6
        nighttime_temp : int, optional
            Goal temperature during nighttime setback, by default 18

        Raises
        ------
        TypeError
            If wrong input types are detected.
        """
        self.controls = control_list

        self.observations = [
            'TOut.T', 'heaPum.COP', 'heaPum.COPCar', 'heaPum.P',
            'heaPum.QCon_flow', 'heaPum.QEva_flow', 'heaPum.TConAct',
            'heaPum.TEvaAct', 'preHea.Q_flow', 'rad.Q_flow', 'rad.m_flow',
            'sunHea.Q_flow', 'sunRad.y', 'temRet.T', 'temRoo.T', 'temSup.T',
            'weaBus.HDifHor', 'weaBus.HDirNor', 'weaBus.HGloHor',
            'weaBus.HHorIR', 'y', 'time']
        self.P = P
        self.set_point = set_point
        self.nighttime_setback = nighttime_setback
        self.nighttime_start = nighttime_start
        self.nighttime_end = nighttime_end
        self.nighttime_temp = nighttime_temp

    def get_control(self, observation, temp_sp, hour=0):
        """Computes the control actions.

        Parameters
        ----------
        obs : dict
            Dict containing the temperature observations.
        temp_sp : float
            Goal temperature for the next timestep.
        hour : int
            Current hour in the simulation time.

        Returns
        -------
        controls : dict
            Dict containing the control inputs.
        """
        controls = {}
        control_temp = self.set_point
        if self.nighttime_setback:
            if hour < self.nighttime_end or hour > self.nighttime_start:
                control_temp = self.nighttime_temp

        control_u = 0.0
        control_u = min(self.P * max(control_temp - observation, 0.0), 1.0)
        controls['u'] = [control_u]

        return controls


class PHouseController(object):
    """Rule-based controller for heat pump control.

    Attributes
    ----------
    controls : list of str
        List of control inputs.
    observations : list of str
        List of zone temperature observations
    tol1 : float
        First threshold for deviation from the goal temperature.
    tol2 : float
        Second threshold for deviation from the goal temperature.
    nighttime_setback : bool
        Whether to use a nighttime setback.
    nighttime_start : int
        Hour to start the nighttime setback.
    nighttime_end : int
        Hour to end the nighttime setback.
    nighttime_temp : float
        Goal temperature during nighttime setback

    Methods
    -------
    get_control(obs, temp_sp, hour)
        Computes the control actions.
    """

    def __init__(
        self,
        control_list,
        P,
        set_point,
        nighttime_setback=False,
        nighttime_start=17,
        nighttime_end=6,
        nighttime_temp=18,
    ):
        """
        Parameters
        ----------
        control_list : list of str
            List containing all inputs
        P : float
            Gain for the P-controller.
        set_point : float
            Daytime temperature set point.
        nighttime_setback : bool, optional
            Whether to use a nighttime setback, by default False.
        nighttime_start : int, optional
            Hour to start the nighttime setback, by default 17
        nighttime_end : int, optional
            Hour to end the nighttime setback, by default 6
        nighttime_temp : int, optional
            Goal temperature during nighttime setback, by default 18

        Raises
        ------
        TypeError
            If wrong input types are detected.
        """
        self.controls = control_list

        self.observations = [
            'TOut.T', 'heaPum.COP', 'heaPum.COPCar', 'heaPum.P',
            'heaPum.QCon_flow', 'heaPum.QEva_flow', 'heaPum.TConAct',
            'heaPum.TEvaAct', 'preHea.Q_flow', 'rad.Q_flow', 'rad.m_flow',
            'sunHea.Q_flow', 'sunRad.y', 'temRet.T', 'temRoo.T', 'temSup.T',
            'weaBus.HDifHor', 'weaBus.HDirNor', 'weaBus.HGloHor',
            'weaBus.HHorIR', 'y', 'time']
        self.P = P
        self.set_point = set_point
        self.nighttime_setback = nighttime_setback
        self.nighttime_start = nighttime_start
        self.nighttime_end = nighttime_end
        self.nighttime_temp = nighttime_temp

    def get_control(self, observation, temp_sp, hour=0):
        """Computes the control actions.

        Parameters
        ----------
        obs : dict
            Dict containing the temperature observations.
        temp_sp : float
            Goal temperature for the next timestep.
        hour : int
            Current hour in the simulation time.

        Returns
        -------
        controls : dict
            Dict containing the control inputs.
        """
        controls = {}
        control_temp = self.set_point
        if self.nighttime_setback:
            if hour < self.nighttime_end or hour > self.nighttime_start:
                control_temp = self.nighttime_temp

        control_u = 0.0
        control_u = min(self.P * max(control_temp - observation, 0.0), 1.0)
        controls['u'] = [control_u]

        return controls


class SimpleHouseController(object):
    """Rule-based controller for heat pump control.

    Attributes
    ----------
    controls : list of str
        List of control inputs.
    observations : list of str
        List of zone temperature observations
    tol1 : float
        First threshold for deviation from the goal temperature.
    tol2 : float
        Second threshold for deviation from the goal temperature.
    nighttime_setback : bool
        Whether to use a nighttime setback.
    nighttime_start : int
        Hour to start the nighttime setback.
    nighttime_end : int
        Hour to end the nighttime setback.
    nighttime_temp : float
        Goal temperature during nighttime setback

    Methods
    -------
    get_control(obs, temp_sp, hour)
        Computes the control actions.
    """

    def __init__(
        self,
        control_list,
        lower_tol,
        upper_tol,
        nighttime_setback=False,
        nighttime_start=17,
        nighttime_end=6,
        nighttime_temp=18,
    ):
        """
        Parameters
        ----------
        control_list : list of str
            List containing all inputs
        lower_tol : float
            First threshold for deviation from the goal temperature.
        upper_tol : float
            Second threshold for deviation from the goal temperature.
        nighttime_setback : bool, optional
            Whether to use a nighttime setback, by default False.
        nighttime_start : int, optional
            Hour to start the nighttime setback, by default 17
        nighttime_end : int, optional
            Hour to end the nighttime setback, by default 6
        nighttime_temp : int, optional
            Goal temperature during nighttime setback, by default 18

        Raises
        ------
        TypeError
            If wrong input types are detected.
        """
        self.controls = control_list

        self.observations = [
            'TOut.T', 'heaPum.COP', 'heaPum.COPCar', 'heaPum.P',
            'heaPum.QCon_flow', 'heaPum.QEva_flow', 'heaPum.TConAct',
            'heaPum.TEvaAct', 'preHea.Q_flow', 'rad.Q_flow', 'rad.m_flow',
            'sunHea.Q_flow', 'sunRad.y', 'temRet.T', 'temRoo.T', 'temSup.T',
            'weaBus.HDifHor', 'weaBus.HDirNor', 'weaBus.HGloHor',
            'weaBus.HHorIR', 'y', 'time']
        self.tol1 = lower_tol
        self.tol2 = upper_tol
        self.nighttime_setback = nighttime_setback
        self.nighttime_start = nighttime_start
        self.nighttime_end = nighttime_end
        self.nighttime_temp = nighttime_temp

    def get_control(self, observation, temp_sp, hour=0):
        """Computes the control actions.

        Parameters
        ----------
        obs : dict
            Dict containing the temperature observations.
        temp_sp : float
            Goal temperature for the next timestep.
        hour : int
            Current hour in the simulation time.

        Returns
        -------
        controls : dict
            Dict containing the control inputs.
        """
        controls = {}
        control_temp = temp_sp
        if self.nighttime_setback:
            if hour < self.nighttime_end or hour > self.nighttime_start:
                control_temp = self.nighttime_temp

        control_u = 0.0
        if (observation - control_temp < self.tol1 and
            control_temp - observation < self.tol1):
            control_u = 0.4
        elif self.tol1 < observation - control_temp < self.tol2:
            control_u = 0.2
        elif observation - control_temp > self.tol2:
            control_u = 0.0
        elif self.tol1 < control_temp - observation < self.tol2:
            control_u = 0.6
        elif control_temp - observation > self.tol2:
            control_u = 0.8
        controls['u'] = [control_u]

        return controls



def get_ApartTherm_kpis(**kwargs):
    weather = "CH_BS_Basel"
    num_sim_days = 30
    env = energym.make(
        "SimpleHouseRad-v0", weather=weather, simulation_days=num_sim_days)
    mins_per_step = 5
    mins_in_a_day = 24 * 60
    steps_in_a_day = int(mins_in_a_day / mins_per_step)

    inputs = env.get_inputs_names()

    if kwargs['controller'] == 'RB':
        lower_tol, upper_tol = kwargs['params']
        controller = SimpleHouseController(
            control_list=inputs, lower_tol=lower_tol, upper_tol=upper_tol,
            nighttime_setback=True, nighttime_start=18, nighttime_end=6,
            nighttime_temp=18
        )
    elif kwargs['controller'] == 'P':
        P, sp = kwargs['params']
        controller = PHouseController(
            control_list=inputs, P=P, set_point=sp
        )
    elif kwargs['controller'] == 'PH':
        P, sp, nighttime_end = kwargs['params']
        controller = PHHouseController(
            control_list=inputs, P=P, set_point=sp, nighttime_end=nighttime_end
        )


    steps = steps_in_a_day * num_sim_days
    out_list = []
    outputs = env.get_output()['temRoo.T']
    hour = 0
    for i in range(steps):
        control = controller.get_control(outputs, 21, hour)
        outputs = env.step(control)['temRoo.T'] - Celcius_to_Kelvin
        _,hour,_,_ = env.get_date()

        out_list.append(outputs)

    kpis = env.get_kpi()
    energy = kpis['kpi1']['kpi']
    avg_dev = kpis['kpi2']['kpi']
    env.close()
    return energy, avg_dev
