""" Integration and Unit tests """

import os
import unittest
import numpy as np
from nimbus import Nimbus, DataBase

# ==== Example values
temperature = np.asarray([775, 951, 1073, 1111, 1540, 2654])  # [K]
pressure = np.asarray([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])  # [bar]
kzz = np.ones_like(pressure) * 1e9  # [cm2/s]
gravity = 10**2.49  # [cm/s2]
mmw = 2.34  # [amu]
species = 'SiO'
deepmmr = 1e-3  # [g/g]

def test_nimbus():
    """ Integration testing """

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/',
                 verbose=True, create_analytic_plots=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=3)
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.00029755903690762727)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.0020267578745118706)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 0.00017738970097126437)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 28.1665833969462)

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=None)
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.00017402989930422082)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.0020261382645410508)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 8.523008406595292e-05)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 18.34792414117775)
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=0)
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.0010772818551208216)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.0020345673147105186)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 0.00012867080982528894)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 161.89495004188024)

    # ==== set up nimbus full
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/', verbose=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.00017415323472720008)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.0020261550284562564)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 1.0313424623047931e-05)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 18.47278993747007)

    # ==== timout test
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full', timeout=0.001)
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 6.000119511527929e-30)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.002034254173417671)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 6.000038568131e-07)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 1.3924126774404749e-18)

    # ==== influx added
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    inj = lambda a, b, c: 1e-11
    obj.set_up_top_of_atmosphere_influx(inj)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.002554693560567451)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.002034855323150379)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 7.880688247351824e-07)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 7602539.859462642)

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='convergence', rel_dif_in_mmr=1e-3, save_file='test')
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.00017411877560450766)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.0020261545373924435)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 8.533026717397001e-05)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 18.467750747986038)

    # ==== load previous run
    ds2 = obj.load_previous_run(ds_prev=ds)
    assert np.isclose(np.sum(np.asarray([ds2['cloud_mmr'][0]]).T), 0.00017411877560450766)
    assert np.isclose(np.sum(np.asarray([ds2['gas_mmr'][0]]).T), 0.0020261545373924435)
    assert np.isclose(np.sum(np.asarray([ds2['cloud_radius']]).T), 8.533026717397001e-05)
    assert np.isclose(np.sum(np.asarray([ds2['cloud_number_density']]).T), 18.467750747986038)
    ds3 = obj.load_previous_run(file_name='test.nc')
    assert np.isclose(np.sum(np.asarray([ds3['cloud_mmr'][0]]).T), 0.00017411877560450766)
    assert np.isclose(np.sum(np.asarray([ds3['gas_mmr'][0]]).T), 0.0020261545373924435)
    assert np.isclose(np.sum(np.asarray([ds3['cloud_radius']]).T), 8.533026717397001e-05)
    assert np.isclose(np.sum(np.asarray([ds3['cloud_number_density']]).T), 18.467750747986038)

    # ==== load previous run
    obj2 = Nimbus(working_dir=os.path.dirname(__file__) + '/working/',
                 verbose=True, create_analytic_plots=True)
    obj2.set_up_from_previous_run(file_name='test.nc')
    assert np.isclose(np.sum(obj2.pres), 1111.11)
    assert np.isclose(np.sum(obj2.temp), 8104)
    assert obj2.isset_initialisation
    obj2.set_up_from_previous_run(ds_prev=ds)
    assert np.isclose(np.sum(obj2.pres), 1111.11)
    assert np.isclose(np.sum(obj2.temp), 8104)
    assert obj2.isset_initialisation
    os.remove('test.nc')

    # ==== set up nimbus with multiple materials
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity,
                          ['SiO', 'MgSiO3'], [1e-3, 1e-4])
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), 0.00028856177206542375)
    assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), 0.002023051669376635)
    assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), 6.0064927387298e-06)
    assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), 181.7379885768039)



def test_solversetters():
    "Unit testing of Nimbus"
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
    """ Unit testing of DataStorage """
    ds = DataBase()
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
    vp = ds.vapor_pressures('Ni', temp+2000)
    assert np.isclose(np.sum(vp), 51228)
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
    vp = ds.monomer_radius('SiO2')
    assert np.isclose(np.sum(vp), 2.079e-08)


def test_spectra():
    """ This function is currently only used for local testing as it relys on
    the MieAi implementation """
    """ Integration testing """
    # ==== Example values
    temperature = np.asarray([775, 951, 1073, 1111, 1540, 2654])  # [K]
    pressure = np.asarray([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])  # [bar]
    kzz = np.ones_like(pressure) * 1e9  # [cm2/s]
    gravity = 10**2.49  # [cm/s2]
    mmw = 2.34  # [amu]
    species = 'MgSiO3'
    deepmmr = 1e-3  # [g/g]

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    obj.compute(typ='iterate', max_iterations=3)
    df_cloud = obj.picaso_formater(mie_type='full', nradii=10)
    assert np.isclose(np.sum(df_cloud['opd']), 355.33030871920846)
    assert np.isclose(np.sum(df_cloud['g0']), 707.2811252692154)
    assert np.isclose(np.sum(df_cloud['w0']), 485.1164743673346)
    assert np.isclose(np.sum(df_cloud['wavenumber']), 6849905.947614839)


def test_asserts():
    # set up testcase class:
    testcase = unittest.TestCase()
    # set up nimbus
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/',
                 verbose=True, create_analytic_plots=True)

    # ==== errors if atmospehre is not setup
    with testcase.assertRaises(ValueError):
        obj.set_up_solver()

    # ==== assertion errors when not at least one input is given
    with testcase.assertRaises(ValueError):
        obj.load_previous_run()
    with testcase.assertRaises(ValueError):
        obj.set_up_from_previous_run()

    # ==== value errors in database
    ds = DataBase()
    temp = np.asarray([500])
    with testcase.assertRaises(ValueError):
        ds.surface_tension('MgO', temp)
    with testcase.assertRaises(ValueError):
        ds.vapor_pressures('MgO', temp)
