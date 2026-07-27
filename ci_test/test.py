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

    # ==== Generic testing function:
    def check(ds, sol):
        print(np.sum(np.asarray([ds['cloud_mmr'][0]]).T))
        print(np.sum(np.asarray([ds['gas_mmr'][0]]).T))
        print(np.sum(np.asarray([ds['cloud_radius']]).T))
        print(np.sum(np.asarray([ds['cloud_number_density']]).T))
        assert np.isclose(np.sum(np.asarray([ds['cloud_mmr'][0]]).T), sol[0])
        assert np.isclose(np.sum(np.asarray([ds['gas_mmr'][0]]).T), sol[1])
        assert np.isclose(np.sum(np.asarray([ds['cloud_radius']]).T), sol[2])
        assert np.isclose(np.sum(np.asarray([ds['cloud_number_density']]).T), sol[3])


    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/',
                 verbose=True, create_analytic_plots=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=3)
    check(ds, [0.00026518302326593995, 0.002027077714200809, 0.00044025281923456183,
               20.893089112469035])

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=None)
    check(ds, [0.00015532806998874973, 0.002026365100614461, 0.00031179687608750683,
               13.807489741627958])
    # change max iterations
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='iterate', max_iterations=0)
    check(ds, [0.0010742212086526902, 0.002042050311914651, 0.00021165110149672897,
               109.33868417230173])

    # ==== set up nimbus full
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/', verbose=True)
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    check(ds, [0.0001550947737573427, 0.0020263800957963235, 1.1334719687814653e-05,
               13.893025318657784])

    # ==== timout test
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full', timeout=0.001)
    check(ds, [6.0000000000027695e-30, 0.002034249820045167, 6e-07,
               1.3924126755600967e-18])

    # ==== influx added
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    inj = lambda a, b, c: 1e-11
    obj.set_up_influx(inj)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    check(ds, [0.001956334984998105, 0.0035141095017851382, 2.750709667060246e-06,
               17082.166474858637])

    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    kzz_f = lambda t, p: (1e9 + 1e2 * t / obj.tend) * np.ones_like(p)
    obj.set_up_atmosphere(temperature, pressure, kzz_f, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='full')
    check(ds, [0.00015509478670362202, 0.0020263800957670604, 1.1334719851846651e-05,
               13.893026038509976])

    # ==== set up nimbus itteratively
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    ds = obj.compute(typ='convergence', rel_dif_in_mmr=1e-3, save_file='test')
    check(ds, [0.00015508052697191544, 0.0020263796685717544, 0.0003225189326803817,
               13.889234016624483])

    # ==== load previous run
    ds2 = obj.load_previous_run(ds_prev=ds)
    check(ds2, [0.00015508052697191544, 0.0020263796685717544, 0.0003225189326803817,
               13.889234016624483])
    ds3 = obj.load_previous_run(file_name='test.nc')
    check(ds3, [0.00015508052697191544, 0.0020263796685717544, 0.0003225189326803817,
               13.889234016624483])

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
    check(ds, [0.00020227752438271322, 0.0020233274149122917, 8.276826302545698e-06,
               74.42661745020483])

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
    assert np.isclose(np.sum(df_cloud['opd']), 319.19585100687664)
    assert np.isclose(np.sum(df_cloud['g0']), 741.0198057441573)
    assert np.isclose(np.sum(df_cloud['w0']), 482.0441774716092)
    assert np.isclose(np.sum(df_cloud['wavenumber']), 6849905.947614839)

    # ==== set up nimbus fully to test timestamps
    obj = Nimbus(working_dir=os.path.dirname(__file__) + '/working/')
    obj.set_up_atmosphere(temperature, pressure, kzz, mmw, gravity, species, deepmmr)
    obj.set_up_solver()
    obj.compute(typ='full')
    df_cloud = obj.picaso_formater(mie_type='full', nradii=10, time_stamps=[1e-2, 1e9])
    sols = [
        [1.2073548471097519e-25, 1677.5748310820318, 195.96456606238877, 6849905.947614839],
        [421.00651392846675, 807.391702601194, 468.78775297265446, 6849905.947614839],
    ]
    for d, df in enumerate(df_cloud):
        assert np.isclose(np.sum(df['opd']), sols[d][0])
        assert np.isclose(np.sum(df['g0']), sols[d][1])
        assert np.isclose(np.sum(df['w0']), sols[d][2])
        assert np.isclose(np.sum(df['wavenumber']), sols[d][3])

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
