""" All set-up functionalities of NIMBUS """

import numpy as np
from scipy.optimize import root_scalar

from .atmosphere_physics import define_atmosphere_physics
from .species_database import DataStorage


def set_up_atmosphere(self, temperature, pressure, kzz, mmw, gravity, fsed, specie, deep_mmr):
    """
    Set up the atmospheric structure of the simulation.

    Parameters
    ----------
    self : Nimbus class
        Nimbus object that is set up
    temperature : np.array
        Temperature in Kelvin.
    pressure : np.array
        Pressure in dyn/cm2.
    kzz : np.array
        Diffusion coefficient in cm2/s.
    mmw : np.array
        Mean molecular weight in amu.
    gravity : np.array
        Gravity in cm/s2
    fsed : np.array
        Initial settling parameter (defines cloud particle size).
    specie: np.array
        Cloud particle specie (currently only 1 is supported).
    deep_mmr: np.array
        Mass mixing ratio of the cloud specie in the deep atmosphere.
    """
    # ==== Setting input parameters
    self.temp = temperature  # temperature profile [K]
    self.pres = pressure  # pressure profile [dyn/cm2]
    self.kzz = kzz  # mixing coefficient [cm2/s]
    self.mmw = mmw  # mean molecular weight [amu]
    self.gravity = gravity  # gravity [cm/s2]
    self.fsed = fsed  # (initial) settling parameter [None]
    self.deep_gas_mmr = deep_mmr  # mass mixing ratio in the interior [g/g]

    # ==== Set nucleation rate, accretion rate, and settling velocity
    define_atmosphere_physics(self)

    # ==== currently hardcoded for SiO, later this will be input
    # specie = 'FeO'
    ds = DataStorage()
    self.r_ccn = 1e-7  # default for minimum cloudparticle radius [cm]
    self.cs_mol = 2e-15  # default for molecular cross section [cm2]
    #   Depth of the Lennard-Jones potential well for the atmosphere
    # Used in the viscocity calculation (units are K) (Rosner, 2000)
    #   (78.6 for air, 71.4 for N2, 59.7 for H2)
    self.eps_k = 59.7
    self.rho_ccn = ds.solid_density(specie)
    self.rhop = ds.solid_density(specie)
    self.mw = ds.molecular_weight(specie)
    self.r1 = ds.monomer_radius(specie)
    self.sig = ds.surface_tension(specie, self.temp)
    self.m1 = ds.monomer_mass(specie)
    self.rgas_spec_cloud = ds.specific_gas_constant(specie)
    self.pvap = ds.vapor_pressures(specie, self.temp)

    # ==== Size and shpae of inputs
    self.sz = len(pressure)

    # ==== calculate pressure grid
    # grid coordiantes
    self.logp = np.log(pressure)  # pressure grid
    self.logp_mid = (self.logp[1:] + self.logp[:-1]) / 2  # midpoints
    # pressure grid bin size
    self.dlogp = np.zeros_like(self.logp)
    self.dlogp[1:-1] = (self.logp[2:] - self.logp[:-2]) / 2
    self.dlogp[0] = self.logp[1] - self.logp[0]
    self.dlogp[-1] = self.logp[-1] - self.logp[-2]
    # midpoints bin size
    self.dlogp_mid = self.logp[1:] - self.logp[:-1]
    # derivatives to be used later
    self.dz = - self.rgas * self.temp / self.mmw / self.gravity * self.dlogp

    # ==== Derive physical properties
    self.natmo = self.pres / self.temp / self.kb  # total gas-phase number density [1/cm3]
    self.rhoatmo = self.mmw * self.pres / self.temp / self.rgas  # atmospheric density [g/cm]
    self.m_ccn = 4 / 3 * np.pi * self.r_ccn ** 3 * self.rho_ccn  # ccn mass [g]
    self.vth = np.sqrt(self.rgas * self.temp / (2 * np.pi * self.mw))  # thermal velocity [cm/s]
    lmfpfac = np.sqrt(2) * self.rhoatmo * self.cs_mol
    self.lmfp = self.mmw / self.avog / lmfpfac  # mean free path [cm]
    self.h = self.rgas * self.temp / self.gravity / self.mmw  # scale height [cm]
    self.ct = np.sqrt(2 * self.rgas * self.temp / self.mmw)  # sound speed [cm/s]

    # ==== mid point values
    self.kzz_mid = np.interp(self.logp_mid, self.logp, self.kzz)
    self.rhoatmo_mid = np.interp(self.logp_mid, self.logp, self.rhoatmo)
    self.temp_mid = np.interp(self.logp_mid, self.logp, self.temp)
    self.dz_mid = - self.rgas * self.temp_mid / self.mmw / gravity * self.dlogp_mid

    # ==== find pressures which are supersaturated
    ndeep = self.deep_gas_mmr * self.rhoatmo / self.m1
    pdeep = ndeep * self.kb * self.temp
    self.mask_psupsat = self.pvap / pdeep < 1

    # ==== working variables
    self.fex = None
    self.j = None
    self.jac = None
    self.x0 = None

    # ==== Default solver settings (can be changed with set_solver_settings())
    self.tstart = 1e-4
    self.tend = 1e10
    self.tsteps = 20
    self.ode_rtol = 1e-3
    self.ode_atol = 1e-25

    # ==== output variabels
    self.rg_history = None

    # ==== Calculate initial radius
    self.rg = np.zeros_like(self.pres)
    for i, _ in enumerate(self.pres):
        # minimisation function
        def vsed_f(rg):
            v_c = self.vsed(rg)[i]  # settling veloctity
            vk = self.fsed * self.kzz[i] / self.h[i]  # fsed velocity
            return vk - v_c

        # call of minimisation function with optimised initial condiaitons
        self.rg[i] = np.maximum(root_scalar(vsed_f, x0=self.r1 * 1e2).root, self.r_ccn)

    # ==== Confirm that atmosphere has been set up
    self.isset_atmosphere = True
