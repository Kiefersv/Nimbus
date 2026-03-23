""" All set-up functionalities of NIMBUS """
# pylint: disable=R0913,E0402,R0915
import numpy as np
from scipy.optimize import root_scalar

from .atmosphere_physics import define_atmosphere_physics
from .species_database import DataStorage

def set_up_atmosphere(self, temperature, pressure, kzz, mmw, gravity, species,
                      deep_mmr, fsed=1, metalicity=1, ignore_as_nucleator=[]):
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
    species: np.array
        Cloud particle specie (currently only 1 is supported).
    deep_mmr: np.array
        Mass mixing ratio of the cloud specie in the deep atmosphere.
    fsed : np.array, optional
        Initial settling parameter (defines cloud particle size).
    metalicity : np.array or float, optional
        metalicity of atmosphere (used for certain pvaps)
    ignore_as_nucleator : List[str]
        Species which should not be considered to nucleate
    """

    # ==== Initialise all cloud species =================================================
    # Note: Each species gets an index according to the input order. Until the output,
    # only the index is used to identify the species.
    if isinstance(species, str):
        self.species = [species]
        self.deep_gas_mmr = np.asarray([deep_mmr])
    elif isinstance(species, (list, tuple)):
        self.species = species
        # if given as dict, transform to ordered list
        if isinstance(deep_mmr, dict):
            deep_mmr = [deep_mmr[spec] for spec in species]
        self.deep_gas_mmr = np.asarray(deep_mmr)

    # ==== Size and shpae of inputs
    self.sz = len(pressure)
    self.nspec = len(self.species)

    # ==== Setting input parameters
    self.temp = temperature  # temperature profile [K]
    self.pres = pressure*1e6  # pressure profile, convert from bar to [dyn/cm2]
    self.kzz = kzz  # mixing coefficient [cm2/s]
    self.mmw = mmw  # mean molecular weight [amu]
    self.gravity = gravity  # gravity [cm/s2]
    self.fsed = fsed  # (initial) settling parameter [None]
    self.mh = metalicity  # metalicity relative to solar (not log!) []
    self.ian = ignore_as_nucleator  # these species will not nucleate

    # ==== Set nucleation rate, accretion rate, and settling velocity
    define_atmosphere_physics(self)

    # ==== currently hardcoded for SiO, later this will be input
    ds = DataStorage()  # open the data storage
    self.ds = ds  # remember the class
    # ==== Assign material information
    # density of cloud material [g/cm3]
    self.rhop = np.asarray([ds.solid_density(spec) for spec in self.species])
    # cloud material molecular weight [amu]
    self.mw = np.asarray([ds.molecular_weight(spec) for spec in self.species])
    # monomer mass [g]
    self.m1 = np.asarray([ds.monomer_mass(spec) for spec in self.species])
    # specific gas constant
    self.rgas_spec_cloud = np.asarray([ds.specific_gas_constant(spec) for spec in self.species])

    # ==== calculate pressure grid
    # grid coordiantes
    self.logp = np.log(self.pres)  # pressure grid
    self.logp_mid = (self.logp[1:] + self.logp[:-1]) / 2  # midpoints
    # pressure grid bin size
    self.dlogp = np.zeros_like(self.logp)
    self.dlogp[1:-1] = (self.logp[2:] - self.logp[:-2]) / 2
    self.dlogp[0] = self.logp[1] - self.logp[0]
    self.dlogp[-1] = self.logp[-1] - self.logp[-2]
    # midpoints bin size
    self.dlogp_mid = self.logp[1:] - self.logp[:-1]

    # ==== Derive physical properties
    self.m_ccn = 4 / 3 * np.pi * self.r_ccn ** 3 * self.rho_ccn  # ccn mass [g]
    self.kzz_mid = np.interp(self.logp_mid, self.logp, self.kzz)

    # ==== pre-compute constant values
    self.calc_atmos_struct()

    # ==== find pressure levels which are always supersaturated
    self.mask_psupsat = self.pres > 0
    for s, spec in enumerate(self.species):
        if self.deep_gas_mmr[s] > 0:
            ndeep = self.deep_gas_mmr[s] * self.rhoatmo / self.m1[s]  # deep particle number density
            pdeep = ndeep * self.kb * self.temp  # deep partial pressure
            pvap = self.ds.vapor_pressures(spec, self.temp, self.mh)
            mask = pvap / pdeep < 1
            self.mask_psupsat *= pvap / pdeep < 1  # mask where vapour can condense

    # ==== Calculate initial radius
    self.rg = np.zeros_like(self.pres)
    for i, _ in enumerate(self.pres):
        # minimisation function
        def vsed_f(rg):
            v_c = self.vsed(rg, self.rho_ccn)[i]  # settling veloctity
            vk = self.fsed * self.kzz[i] / self.h[i]  # fsed velocity
            return vk - v_c
        # call of minimisation function with optimised initial condiaitons
        self.rg[i] = np.maximum(root_scalar(vsed_f, x0=self.r1 * 1e2).root, self.r_ccn)

    # ==== Confirm that atmosphere has been set up
    self.isset_atmosphere = True

    # ==== Print current setup
    if not self.mute:
        print('[INFO] Atmosphere set up with:')
        print(f'       -> pressure range: {np.max(pressure):.2e} - {np.min(pressure):.2e} bar')
        print(f'       -> temperature range: {np.max(self.temp):.2e} - {np.min(self.temp):.2e} K')
        print(f'       -> Kzz range: {np.max(kzz):.2e} - {np.min(kzz):.2e} cm2/s')
        print(f'       -> Mean molecular weight: {mmw:.2e} amu')
        print(f'       -> Gravity: {gravity:.2e} cm/s2')
        for s in range(self.nspec):
            print('       -> ' + self.species[s] + f' deep MMR: {self.deep_gas_mmr[s]:.2e} g/g')


def set_up_top_of_atmosphere_influx(self, influx_function):
    """
    Set up the top of atmosphere source function.

    Parameters
    ----------
    self : Nimbus class
        Nimbus object that is set up
    influx_function : Function
        def top_function(pressure, temperature, time):
            pressure : np.ndarray[N]
                pressure structure
            temperature : np.ndarray[N]
                temperature structure
            time : float
                current time (can be unused if constant)
            return : np.ndarray[3, N]
                the influx of gas-phase material at index 0, solid cloud material at
                index 1, and cloud particles at index 2. Index 1 should in general be
                all zeros.
    """
    self.tf = influx_function

    # ==== Print current setup
    if not self.mute:
        print('[INFO] Top of atmosphere influx function added')

def calc_atmos_struct(self):
    """ This function performs atmospheric calculation updates """

    # ==== Derive physical properties
    self.natmo = self.pres / self.temp / self.kb  # total gas-phase number density [1/cm3]
    self.rhoatmo = self.mmw * self.pres / self.temp / self.rgas  # atmospheric density [g/cm]
    self.vth = np.sqrt(8 * self.rgas * self.temp / (np.pi * self.mmw))
    lmfpfac = np.sqrt(2) * self.rhoatmo * self.cs_mol
    self.lmfp = self.mmw / self.avog / lmfpfac  # mean free path length [cm]
    self.h = self.rgas * self.temp / self.gravity / self.mmw  # scale height [cm]
    self.ct = np.sqrt(2 * self.rgas * self.temp / self.mmw)  # sound speed [cm/s]
    # derivatives to be used later
    self.dz = - self.rgas * self.temp / self.mmw / self.gravity * self.dlogp

    # ==== mid point values (see above for explenation of values)
    self.rhoatmo_mid = np.interp(self.logp_mid, self.logp, self.rhoatmo)
    self.temp_mid = np.interp(self.logp_mid, self.logp, self.temp)
    self.dz_mid = - self.rgas * self.temp_mid / self.mmw / self.gravity * self.dlogp_mid
