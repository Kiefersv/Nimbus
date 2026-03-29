""" This is the header file of Nimbus. To find functionalities check the src folder. """
# pylint: disable=C0415

class Nimbus:
    """ Main class to handle the cloud structure calculation """

    # ==== physical constants
    rgas = 8.3143e7  # universal gas constant [erg/mol/K]
    avog = 6.02e23  # Avogadro constant [mol]
    kb = rgas / avog  # boltzmann constant [erg/K]
    pf = 1000000  # Reference pressure [dyn/cm2]

    # ==== import set up functions
    from .src.atmosphere import set_up_atmosphere
    from .src.atmosphere import set_up_top_of_atmosphere_influx
    from .src.atmosphere import calc_atmos_struct
    from .src.solver import set_up_solver

    # ==== import the main compuational function
    from .src.compute import compute

    # ==== import solver settings
    from .src.settings import set_cloud_settings, set_solver_settings, set_fudge_settings

    # ==== import data handling functions
    from .src.data_storage import load_previous_run
    from .src.data_storage import set_up_from_previous_run

    # ==== import plotting routines
    from .src.spectra import picaso_formater

    def __init__(self, working_dir='.', create_analytic_plots=False, verbose=False,
                 mute=False):
        """
        Constructor.

        Parameters
        ----------
        working_dir : str, optional
            Directory to store and load files from
        create_analytic_plots : bool, optional
            If true, analytic plots will be produced, increases computation time
        verbose : bool, optional
            If true, additional information are printed, increases computation time
        mute : bool, optional
            If true, Nimbus runs completely quietly, no output at all.
        """

        # ==== Workflow settings
        self.working_dir = working_dir + '/'  # working directory
        self.do_plots = create_analytic_plots  # If true, analytic plots are created
        self.verbose = verbose  # print diagonsic infos on the cost of computation time
        self.mute = mute  # overwrites verbose and do_plots and makes Nimbus run quietly

        # ==== Default solver settings (can be changed with set_solver_settings())
        self.tstart = 1e-4  # start time of simulation [s]
        self.tend = 1e12  # end time of simulation [s]
        self.tsteps = 20  # number of intermediated evaluations (log-spaced)
        self.ode_rtol = 1e-3  # relative error of solve_ivp
        self.ode_atol = 1e-25  # absolute error of solve_ivp
        self.ode_minimum_mmr = 1e-30  # lowest MMR considered [g/g]
        self.static_rg = True # True: itarate with const rg / False: calc rg on the fly

        # ==== Default cloud physics parameters
        self.r_ccn = 1e-7  # default for minimum cloudparticle radius [cm]
        self.cs_mol = 2e-15  # default for molecular cross section [cm2]
        self.eps_k = 59.7  # Depth of the Lennard-Jones potential [??]
        self.r1 = 2.001e-8 # monomer radius [cm]
        self.rho_ccn = 2.18  # density of nucleation seads [g/cm3]
        self.rg_fit_deg = 8  # degree of the polynomial to fit the ittarative radius

        # ==== storage variables
        self.results = {}  # stores results
        self.rg_history = None  # stores history of rg.
        self.loop_nr = 0  # remember the current loop number
        self.itterations = 0  # remember number of itterations performed

        # ==== Fudge factors to play around with simulation
        # NOTE: The default values here correspond to a non-fudged run
        self.nuc_rate_fudge = 1  # factor to reduce or increase nucleation rate
        self.sticking_coefficient = 1  # of collisional accreatin reaction rates

        # ==== initialisation checks
        self.isset_atmosphere = False  # checks if an atmosphere was initialised
        self.isset_solver = False  # checks if solver is set up
        self.isset_initialisation = False  # checks if initial condistions are set up
        self.isset_transmission_spectrum = False  # checks if all info for ts are given
        self.isset_emission_spectrum = False  # checks if all info for es are given

        # ==== working variables
        self.fex = None  # right hand side of time evolution (solved with solve_ivp)
        self.jac = None  # Jacobian matrix [currently not used]
        self.x0 = None  # initial mass mixing ratios [g/g]
        self.tf = None  # top flux function (see setup function for details)
        self.ds = None  # Internal DataBase object
        self.evaltimes = None  # evaluation timesteps
        self.timeout = None  # time after which the solver is stopped [s]
        self.tfailed = None  # set to eval time when computation failed [s]
        self.start_time = None  # time when compute was started [s]
        self.complete = True  # only set false if computation had to be stopped
        self.yin_store = None  # initial condition storage

        # ==== Atmospheric parameters
        self.temp = None  # temperature profile [K]
        self.pres = None  # pressure profile, convert from bar to [dyn/cm2]
        self.kzz = None  # mixing coefficient [cm2/s]
        self.mmw = None  # mean molecular weight [amu]
        self.gravity = None  # gravity [cm/s2]
        self.fsed = None  # (initial) settling parameter [None]
        self.mh = None  # metalicity relative to solar (not log!) []
        self.ian = None  # these species will not nucleate
        self.rhop = None # density of cloud material [g/cm3]
        self.mw = None  # cloud material molecular weight [amu]
        self.m1 = None  # monomer mass [g]
        self.rg = None  # cloud particle radius [cm]
        self.rgas_spec_cloud = None  # specific gas constant
        self.logp = None  # pressure in log10([dyn/cm2])
        self.logp_mid = None  # mid pressure levels in log10([dyn/cm2)]
        self.dlogp = None  # pressure grid bin size in log10([dyn/cm2)]
        self.dlogp_mid = None  # mid pressure grid bin size in log10([dyn/cm2)]
        self.kzz_mid = None  # mid Kzz levels in log10([cm2/s)]
        self.m_ccn = None  # CCN mass, derived from r_ccn and rho_ccn [g]
        self.mask_psupsat = None  # mask of computational domain
        self.natmo = None  # total gas-phase number density [1/cm3]
        self.rhoatmo = None  # atmospheric density [g/cm]
        self.vth = None  # thermal velocity [cm/s]
        self.lmfp = None  # mean free path length [cm]
        self.h = None  # scale height [cm]
        self.ct = None  # sound speed [cm/s]
        self.dz = None  # altitude difference between alyers [cm]
        self.rhoatmo_mid = None  # atmospheric density at mid-pressure [g/cm3]
        self.temp_mid = None  # temperature at mid-pressure [K]
        self.dz_mid = None  # altitude bin zise at mid pressure [cm]

        # ==== Welcom message
        if not self.mute:
            print('===========================================================')
            print('                   Welcome to Nimbus                       ')
            print('===========================================================')
            print('[INFO] For questions contact: kiefersv.mail@gmail.com')
            print('[INFO] Settings selected:')
            print('       -> working directory: ' + self.working_dir[:-1])
            print('       -> verbose: ' + str(verbose))
            print('       -> analytic plots: ' + str(create_analytic_plots))
