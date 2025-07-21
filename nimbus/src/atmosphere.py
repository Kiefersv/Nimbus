""" All set-up functionalities of NIMBUS """

import numpy as np
from scipy.optimize import root_scalar

from .atmosphere_physics import (define_atmosphere_physics, get_nucleation_rate_function,
                                 get_accretion_rate_function)
from .species_database import DataStorage

# ==== file variables
# namer for accreation rates
acc_rec_nr = 1

def set_up_atmosphere(self, temperature, pressure, kzz, mmw, gravity, metalicity=1,
                      fsed=1, mixed_clouds=True):
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
    metalicity : np.array or float, optional
        metalicity of atmosphere (used for certain pvaps), not always used
    fsed : np.array, optional
        Initial settling parameter (defines cloud particle size).
    mixed_clouds : bool, optional
        If true, cloud particles are assumed to be mixed.
    """

    # ==== Size and shpae of inputs
    self.sz = len(pressure)

    # ==== Setting input parameters
    self.temp = temperature  # temperature profile [K]
    self.pres = pressure  # pressure profile [dyn/cm2]
    self.kzz = kzz  # mixing coefficient [cm2/s]
    self.mmw = mmw  # mean molecular weight [amu]
    self.gravity = gravity  # gravity [cm/s2]
    self.fsed = fsed  # (initial) settling parameter [None]
    self.mh = metalicity  # metalicity relative to solar (not log!) [None]
    self.mixed = mixed_clouds  # if true, clouds are mixed

    # ==== Set nucleation rate, accretion rate, and settling velocity
    define_atmosphere_physics(self)

    # ==== currently hardcoded for SiO, later this will be input
    ds = DataStorage()  # open the data storage
    self.datastorage = ds  # remember the class
    self.r_ccn = 1e-7  # default for minimum cloudparticle radius [cm]
    self.cs_mol = 2e-15  # default for molecular cross section [cm2]
    self.eps_k = 59.7  # Depth of the Lennard-Jones potential [??]
    self.rho_ccn = 3  # density of nucleation seads [g/cm3] (does not impact results)
    self.m_ccn = 4/3 * np.pi * self.r_ccn**3 * self.rho_ccn  # monomer mass [g]

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
    lmfpfac = np.sqrt(2) * self.rhoatmo * self.cs_mol
    self.lmfp = self.mmw / self.avog / lmfpfac  # mean free path length [cm]
    self.h = self.rgas * self.temp / self.gravity / self.mmw  # scale height [cm]
    self.ct = np.sqrt(2 * self.rgas * self.temp / self.mmw)  # sound speed [cm/s]

    # ==== mid point values (see above for explenation of values)
    self.kzz_mid = np.interp(self.logp_mid, self.logp, self.kzz)
    self.rhoatmo_mid = np.interp(self.logp_mid, self.logp, self.rhoatmo)
    self.temp_mid = np.interp(self.logp_mid, self.logp, self.temp)
    self.dz_mid = - self.rgas * self.temp_mid / self.mmw / gravity * self.dlogp_mid

    # ==== Confirm that atmosphere has been set up
    self.isset_atmosphere = True
    print(f'[INFO] Atmosphere set up with:')
    print(f'       -> pressure range: {np.max(pressure*1e-6):.2e} - {np.min(pressure*1e-6):.2e} bar')
    print(f'       -> temperature range: {np.max(temperature):.2e} - {np.min(temperature):.2e} K')
    print(f'       -> Kzz range: {np.max(kzz):.2e} - {np.min(kzz):.2e} cm2/s')
    print(f'       -> Mean molecular weight: {mmw:.2e} amu')
    print(f'       -> Gravity: {gravity:.2e} cm/s2')

def set_initial_mmr(self, mmr):
    """
    Do what the name says and set the initial mmrs.

    mmr: dict
        Dictionary of mmrs {'SiO': 3e-4, 'SiO[s]': 0}
    """

    # if no species were given, quit
    if mmr is None:
        return

    # loop over all species
    for spec in mmr:
        # add new species
        if spec not in self.species:
            self.species.append(spec)
            self.initial_mmrs.append(mmr[spec])
        # update old species
        else:
            self.initial_mmrs[self.species.index(spec)] = mmr[spec]

def add_nucleation_species(self, nucleation_species, initial_mmr=None):
    """
    Add a new nucleation species and all needed parameters

    nucleation_species : string
        Name of gas-phase species which nucleates
    initial_mmr: dict, optional
        Mass mixing ratio of the cloud specie in the deep atmosphere.
    """

    # ==== check if nucleating species already exists
    if nucleation_species not in self.species:
        self.species.append(nucleation_species)
    if nucleation_species + '[s]' not in self.species:
        self.species.append(nucleation_species + '[s]')
        self.idl_clmat.append(len(self.species)-1)
        self.idl_vsed.append(len(self.species)-1)

    # ==== get data of nuclating species
    # vapor pressure function
    pvap = self.datastorage.vapor_pressure_function(nucleation_species)
    r1 = self.datastorage.monomer_radius(nucleation_species)
    sig = self.datastorage.surface_tension_function(nucleation_species)
    mw = self.datastorage.molecular_weight(nucleation_species)

    # ==== get the vapor pressure function and the nucleation rate
    nuc_rate = get_nucleation_rate_function(self, r1, sig, pvap, mw)

    # ==== save the nucleation rate
    self.nuc_reacs[nucleation_species] = {'k': nuc_rate, 'i':nucleation_species,
                                          'mw': mw}

    # ==== set mmrs
    set_initial_mmr(self, initial_mmr)

def add_accreation_reaction(self, cloud_species, gas_phase_reactants,
                            gas_phase_products, initial_mmr=None):
    """
    Add a new cloud formation reaction and all needed parameters

    cloud_species : string
        Name of the cloud particle material
    gas_phase_reactants: List
        Condensing gas-phase species (can be 1 or multiple)
    gas_phase_products: List
        Condensing gas-phase species (can be 0 or multiple)
    initial_mmr: dict, optional
        Mass mixing ratio of the cloud specie in the deep atmosphere.
    """

    # ==== check if all involved species already exists
    all = [cloud_species] + gas_phase_products + gas_phase_reactants
    for spec in all:
        if spec not in self.species:
            self.species.append(spec)
            self.idl_clmat.append(len(self.species)-1)
            self.idl_vsed.append(len(self.species)-1)

    # ==== find and remember molecular weights
    mwr = []
    for spec in gas_phase_reactants:
        mwr.append(self.datastorage.molecular_weight(spec))
    mwp = []
    for sepc in gas_phase_products:
        mwp.append(self.datastorage.molecular_weight(sepc))

    # ==== get data of accreting species
    # vapor pressure function
    pvap = self.datastorage.vapor_pressure_function(cloud_species[:-3])
    r1 = self.datastorage.monomer_radius(cloud_species[:-3])
    mw = self.datastorage.molecular_weight(cloud_species[:-3])

    # ==== get the vapor pressure function and the nucleation rate
    acc_rate = get_accretion_rate_function(self, r1, pvap, mw)

    # ==== save the nucleation rate
    name = 'reac_nr_' + str(acc_rec_nr)
    self.nuc_reacs[name] = {'k': acc_rate, 'i': gas_phase_reactants,
                            'o': gas_phase_products, 'mwr': mwr, 'mwp': mwp}

    # ==== set mmrs
    set_initial_mmr(self, initial_mmr)