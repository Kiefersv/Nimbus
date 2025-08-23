import os
import numpy as np
from nimbus import Nimbus, DataStorage

def test_nimbus():
    # ==== Example values
    # temperature = np.asarray([554, 572, 607, 653, 775, 951, 1073, 1111, 1540, 2654])
    # pressure = np.asarray([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    temperature = np.asarray([775, 951, 1073, 1111, 1540, 2654])
    pressure = np.asarray([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    kzz = np.ones_like(pressure) * 1e9
    gravity = 10**2.49
    mmw = 2.34
    species = 'SiO'
    deepmmr = 1e-3
    fsed = 1

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir='working/', verbose=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, fsed, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(type='iterate', max_itterations=3)
    y = np.asarray([ds['qc']]).T
    assert np.isclose(np.sum(y), 4.855594052733314e-05)

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir='working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, fsed, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(type='convergence', rel_dif_in_mmr=1e-3, save_file='test')
    y = np.asarray([ds['qc']]).T
    assert np.isclose(np.sum(y), 1.4858344032895963e-05)

    # ==== load previous run
    ds = obj.load_previous_run('test.nc')
    y = np.asarray([ds['qc']]).T
    assert np.isclose(np.sum(y), 1.4858344032895963e-05)
    os.remove('test.nc')

    # ==== simple breack down test

def test_solversetters():
    obj = Nimbus(working_dir='working/')

    obj.set_solver_settings(initial_time_for_solver=1, end_time_for_solver=2,
        evaluation_steps_for_solver=3, degree_of_radius_polinomial=4)
    assert obj.tstart == 1
    assert obj.tend == 2
    assert obj.tsteps == 3
    assert obj.rg_fit_deg == 4

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
