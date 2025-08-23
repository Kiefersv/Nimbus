import os
import numpy as np
from nimbus import Nimbus

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
    obj = Nimbus(working_dir='working/')
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
