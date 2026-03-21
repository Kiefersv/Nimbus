""" Integration and Unit tests """

import os
import numpy as np
from nimbus import Nimbus, DataStorage

def test_nimbus():
    # ==== Example values
    temperature = np.asarray([775, 951, 1073, 1111, 1540, 2654])  # [K]
    pressure = np.asarray([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])  # [bar]
    kzz = np.ones_like(pressure) * 1e9  # [cm2/s]
    gravity = 10**2.49  # [cm/s2]
    mmw = 2.34  # [amu]
    species = 'SiO'
    deepmmr = 1e-3  # [g/g]

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/', verbose=True, create_analytic_plots=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=3)
    y = np.asarray([ds['cloud_mmr'][0, -1]]).T
    assert np.isclose(np.sum(y), 0.00029755903690762727)

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    y = np.asarray([ds['cloud_mmr'][0, -1]]).T
    assert np.isclose(np.sum(y), 0.00017415323472720008)

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='convergence', rel_dif_in_mmr=1e-3, save_file='test')
    y = np.asarray([ds['cloud_mmr'][0, -1]]).T
    assert np.isclose(np.sum(y), 0.00017411877560450766)

    # ==== load previous run
    ds = obj.load_previous_run('test.nc')
    y = np.asarray([ds['cloud_mmr'][0, -1]]).T
    assert np.isclose(np.sum(y), 0.00017411877560450766)
    os.remove('test.nc')

    # ==== set up nimbus with multiple materials
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity,
                          ['SiO', 'MgSiO3'], [1e-3, 1e-4])
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    y = np.asarray([ds['cloud_mmr'][0, -1]]).T
    assert np.isclose(np.sum(y), 0.99)



def test_solversetters():
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')

    obj.set_solver_settings(initial_time_for_solver=1, end_time_for_solver=2,
        evaluation_steps_for_solver=3, degree_of_radius_polinomial=4, rtol=5, atol=6,
        ode_minimum_mmr=7)
    assert obj.tstart == 1
    assert obj.tend == 2
    assert obj.tsteps == 3
    assert obj.rg_fit_deg == 4
    assert obj.ode_rtol == 5
    assert obj.ode_atol == 6
    assert obj.ode_minimum_mmr == 7

    obj.set_cloud_settings(minimum_cloud_particle_radius=1, molecular_cross_section=2)
    assert obj.r_ccn == 1
    assert obj.cs_mol == 2

    obj.set_fudge_settings(nucleation_rate_fudge=1, accreation_rate_fudge=1,
                           sticking_coefficient=1)
    assert obj.nuc_rate_fudge == 1
    assert obj.sticking_coefficient == 1

def test_datastorage():
    ds = DataStorage()
    temp = np.asarray([500])
    vp = ds.vapor_pressures('C', 3500)
    assert np.isclose(np.sum(vp), 3200)
    vp = ds.vapor_pressures('CH4', temp)
    assert np.isclose(np.sum(vp), 1266411405)
    vp = ds.vapor_pressures('Fe', temp+3000)
    assert np.isclose(np.sum(vp), 474395)
    vp = ds.vapor_pressures('H2O', temp)
    assert np.isclose(np.sum(vp), 24544251)
    vp = ds.vapor_pressures('H2S', temp)
    assert np.isclose(np.sum(vp), 407030596)
    vp = ds.vapor_pressures('S2', temp)
    assert np.isclose(np.sum(vp), 6)
    vp = ds.vapor_pressures('S8', temp)
    assert np.isclose(np.sum(vp), 4427)
    vp = ds.vapor_pressures('SiO2', temp+1000)
    assert np.isclose(np.sum(vp), 2.111867499419599)
    vp = ds.vapor_pressures('KCl', 3500)
    assert np.isclose(np.sum(vp), 2.880546279845562e+23)
    vp = ds.gibbs_free_energy('SiO2', 1000)
    assert np.isclose(np.sum(vp), -9854695640143.047)
