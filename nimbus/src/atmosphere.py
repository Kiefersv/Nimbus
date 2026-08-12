""" All set-up functionalities of NIMBUS """
# pylint: disable=R0913,E0402,R0915
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from .atmosphere_physics import define_atmosphere_physics
from .species_database import DataBase
from .subfunctions import aoftf

def set_up_atmosphere(self, temperature, pressure, kzz, mmw, gravity, species=None,
                      deep_mmr=None, fsed=1, metalicity=1, ignore_as_nucleator=[]):
    """
    Set up the atmospheric structure of the simulation.

    Parameters
    ----------
    self : Nimbus class
        Nimbus object that is set up
    temperature : np.array
        Temperature in Kelvin.
    pressure : np.array
        Pressure in bar.
    kzz : np.array or function kzz(p, t)
        Diffusion coefficient in cm2/s.
    mmw : np.array
        Mean molecular weight in amu.
    gravity : np.array
        Gravity in cm/s2
    species: np.array, optional
        Cloud particle specie (currently only 1 is supported).
    deep_mmr: np.array, optional
        Mass mixing ratio of the cloud specie in the deep atmosphere.
    fsed : np.array, optional
        Initial settling parameter (defines cloud particle size).
    metalicity : np.array or float, optional
        metalicity of atmosphere (used for certain pvaps)
    ignore_as_nucleator : List[str]
        Species which should not be considered to nucleate
    """

    # ==== Open a database
    db = DataBase()  # open the data storage

    # ==== Check if species are given, if not, calculate internally ====================
    if species is None:
        if not self.mute:
            print('[WARN] Species are set automatically. This is not recommended. '
                  'Please use the list of species provided as a starting point to '
                  'curate your own list.')
        species = self.find_cloud_species(
            temperature, pressure, mmw=mmw, metalicity=metalicity, verbose=self.mute
        )
        deep_mmr = np.asarray([db.solar_mmr(spec, metalicity) for spec in self.species])
    else:
        if deep_mmr is None:
            raise ValueError("[ERROR] Deep MMR is missing.")

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
    self.kzz = aoftf(kzz)  # mixing coefficient [cm2/s]
    self.mmw = mmw  # mean molecular weight [amu]
    self.gravity = gravity  # gravity [cm/s2]
    self.fsed = fsed  # (initial) settling parameter [None]
    self.mh = metalicity  # metalicity relative to solar (not log!) []
    self.ian = ignore_as_nucleator  # these species will not nucleate

    # ==== Set nucleation rate, accretion rate, and settling velocity
    define_atmosphere_physics(self)

    # ==== read in information from species database
    self.db = db  # remember the class
    # ==== Assign material information
    # density of cloud material [g/cm3]
    self.rhop = np.asarray([db.solid_density(spec) for spec in self.species])
    # cloud material molecular weight [amu]
    self.mw = np.asarray([db.molecular_weight(spec) for spec in self.species])
    # monomer mass [g]
    self.m1 = np.asarray([db.monomer_mass(spec) for spec in self.species])
    # specific gas constant
    self.rgas_spec_cloud = np.asarray([db.specific_gas_constant(spec) for spec in self.species])

    # ==== calculate pressure grid
    # grid coordiantes
    self.logp = np.log(self.pres)  # pressure grid in natural logarithm
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
    def kzz_mid(t, p):
        """ Kzz at log midpoint value """
        _kzz = self.kzz(t, p)
        return (_kzz[1:] + _kzz[:-1])/2
    self.kzz_mid = kzz_mid

    # ==== pre-compute constant values
    self.calc_atmos_struct()

    # ==== find pressure levels which are supersaturated
    self.mask_sat = np.zeros((len(species)+1, self.sz), dtype=bool)  # mask to evaporate nc
    # immediately evaporate all cloud particles below the cloud
    for s, spec in enumerate(self.species):
        # calculate vapour pressure curve
        pvap = self.db.vapor_pressures(spec, self.temp, self.mh)
        # calculate partial pressure
        n1 = self.deep_gas_mmr[s] * self.rhoatmo / self.m1[s]  # deep particle number density
        p1 = n1 * self.kb * self.temp  # deep partial pressure
        self.mask_sat[s] = p1 / pvap >= 1  # mask where vapour can condense
        # updated the below cloud mask
        self.mask_sat[-1] += self.mask_sat[s]

    # ==== Calculate initial radius
    self.rg = np.zeros_like(self.pres)
    for i, _ in enumerate(self.pres):
        # minimisation function
        def vsed_f(rg):
            v_c = self.vsed(rg, self.rho_ccn)[i]  # settling veloctity
            vk = self.fsed * self.kzz(0, self.pres)[i] / self.h[i]  # fsed velocity
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
        kval = self.kzz(0, self.pres)
        print(f'       -> Kzz range at t=0: {np.max(kval):.2e} - {np.min(kval):.2e} cm2/s')
        print(f'       -> Mean molecular weight: {mmw:.2e} amu')
        print(f'       -> Gravity: {gravity:.2e} cm/s2')
        for s in range(self.nspec):
            print('       -> ' + self.species[s] + f' deep MMR: {self.deep_gas_mmr[s]:.2e} g/g')


def set_up_influx(self, influx_function):
    """
    Set up the top of atmosphere source function.

    Parameters
    ----------
    self : Nimbus class
        Nimbus object that is set up
    influx_function : Function
        def top_function(nimbus, pressure, temperature, time):
            nimbus : Nimbus class
                Current nimbus class
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

    # ==== mid-point values (see above for explenation of values)
    self.rhoatmo_mid = np.interp(self.logp_mid, self.logp, self.rhoatmo)
    self.temp_mid = np.interp(self.logp_mid, self.logp, self.temp)
    self.dz_mid = - self.rgas * self.temp_mid / self.mmw / self.gravity * self.dlogp_mid

def _find_cloud_species(temperature, pressure, species=None, mmw=2.34,
                        mmr_cloudspecies=1e-2, metallicity=1, verbose=True,
                        create_analytic_plots=True, plot_save_file=None):
    """
    This function returns a list of species that should be considered as
    condensation species.

    Parameters
    ----------
    temperature : np.ndarray[N]
        temperature structure of the atmosphere [K]
    pressure : np.ndarray[N]
        pressure structure of the atmosphere [dyn/cm2]
    species : list[str], optional
        species to consider, if None, all available species are used
    mmw : float, optional
        mean molecular weight of the atmosphere
    mmr_cloudspecies : float or np.ndarray[N], optional
        cloud species mmr. An upper estimate is useful here.
    metallicity : float, optional
        metallicity of the atmosphere
    verbose : bool, optional
        If true, prints about the species selected will be produced
    create_analytic_plots : bool, optional
        If true, produces a plot showing the condensation temperature curves
    plot_save_file : str, None
        If None, plot is shown, if file name given, plot is saved under that name

    Return
    ------
    species_out : List[str]
        list of species that should be considered
    """

    # ==== Initialisation
    # information
    if verbose:
        print(f'[INFO] The following cloud species might form clouds:')
    # physical constants
    rgas = 8.3143e7  # universal gas constant [erg/mol/K]
    avog = 6.02e23  # Avogadro constant [mol]
    kb = rgas / avog  # boltzmann constant [erg/K]
    # database of thermodynamic data of the cloud forming species
    db = DataBase()  # open the data storage
    rhoatmo = mmw * pressure / temperature / rgas
    # check all species if not any specific is given
    if species is None:
        species = db.list_complete_species()
    # empty list for output
    species_out = []
    # check if specific mmrs are given or just a float
    if isinstance(mmr_cloudspecies, float):
        mmr = [mmr_cloudspecies for _ in species]
    else:
        mmr = mmr_cloudspecies

    # ==== Check species
    for s, spec in enumerate(species_out):
        pvap = db.vapor_pressures(spec, temperature, metallicity)
        pvap = np.maximum(pvap, 1e-200)
        # partial pressure of cloud forming species
        m1 = db.monomer_mass(spec)  # mass of monomer
        p1 =  mmr[s] * rhoatmo / m1 * kb * temperature
        # saturation ratio
        s = p1 / pvap
        # add species if it is above at any point in the atmosphere above 1 (so it
        # can condense) and below 1 (so it is not already fully condensed throughout
        # the atmosphere)
        if (s >= 1).any() and (s <= 1.0).any():
            species_out.append(spec)
            if verbose:
                print(f'       -> {spec}')

    # ==== if analytic plots are anabled, print the structure
    if create_analytic_plots:
        # plotting style
        fig, ax = plt.subplots(1, 1)
        ax.set_yscale('log')
        ax.set_ylim(pressure[-1], pressure[0])
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [Bar]')
        # plot the tp-profile
        ax.plot(temperature, pressure, color='k')
        # plot temperature curves for saturation of species
        for s, spec in enumerate(species):
            tvap = db.condensation_temperature(spec, pressure, metallicity, mmw, mmr[s])
            ax.plot(tvap, pressure, label=spec, linestyle='--')
        ax.legend()
        # either save or show plot
        if plot_save_file is not None:
            fig.savefig(plot_save_file)
        else:
            plt.show()

    # ==== return the list of possible species
    return species_out

def find_cloud_species(temperature, pressure, species=None, mmw=2.34,
                        mmr_cloudspecies=1e-2, metallicity=1, verbose=True,
                        create_analytic_plots=True, plot_save_file=None):
    """
    Outside wrapper function that converts pressure to cgs.
    See _find_clod_species() for more details.
    """
    return _find_cloud_species(
        temperature, pressure*1e6, species, mmw, mmr_cloudspecies, metallicity, verbose,
        create_analytic_plots, plot_save_file
    )
