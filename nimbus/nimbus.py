""" This is the header file of Nimbus. To find functionalities check the src folder. """

class Nimbus:
    """ Main class to handle the cloud structure calculation """

    # ==== physical constants
    rgas = 8.3143e7  # universal gas constant [erg/mol/K]
    avog = 6.02e23  # Avogadro constant [mol]
    kb = rgas / avog  # boltzmann constant [erg/K]
    pf = 1000000  # Reference pressure [dyn/cm2]

    # ==== import set up functions
    from .src.atmosphere import (set_up_atmosphere, set_initial_mmr,
                                 add_nucleation_species, add_accreation_reaction)
    from .src.solver import set_up_solver
    from .src.spectra import set_up_spectra_calculation

    # ==== import the main compuational function
    from .src.compute import compute
    from .src.compute_ddw import compute_ddw

    # ==== import solver settings
    from .src.settings import set_cloud_settings, set_solver_settings, set_fudge_settings

    # ==== import data handling functions
    from .src.data_storage import load_previous_run

    # ==== import plotting routines
    from .src.spectra import plot_spectrum

    def __init__(self, working_dir='.', create_analytic_plots=False, verbose=False):
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
        """

        # ==== Workflow settings
        self.working_dir = working_dir + '/'  # working directory
        self.do_plots = create_analytic_plots  # If true, analytic plots are created
        self.verbose = verbose  # print diagonsic infos on the cost of computation time

        # ==== initialisation checks
        self.isset_atmosphere = False  # checks if an atmosphere was initialised
        self.isset_solver = False  # checks if solver is set up
        self.isset_transmission_spectrum = False  # checks if all info for ts are given
        self.isset_emission_spectrum = False  # checks if all info for es are given

        # ==== working variables of solve_ivp
        self.fex = None  # right hand side of time evolution (solved with solve_ivp)
        self.jac = None  # Jacobian matrix, currently not used
        self.x0 = None  # initial mass mixing ratios

        # ==== Default solver settings (can be changed with set_solver_settings())
        self.tstart = 1e-4  # start time of simulation [s]
        self.tend = 1e15  # end time of simulation [s]
        self.tsteps = 20  # number of intermediated evaluations (log-spaced)
        self.ode_rtol = 1e-6  # relative error of solve_ivp
        self.ode_atol = 1e-25  # absolute error of solve_ivp
        self.ode_minimum_mmr = 1e-30  # lowest MMR considered [g/g]
        self.static_rg = True # True: itarate with const rg / False: calc rg on the fly

        # ==== Solver working variables
        self.species = ['xn']  # list of species, index is their internal id
        self.initial_mmrs = [0]  # list of initial mmrs, same index as self.species
        # dictionary of chemical reactions, example:
        # {"name": {'k':f(temp, pres), 'i': ['Si', 'O2'], 'o': ['SiO', 'O']}}
        self.chem_reacs = {}
        # dictionary of nucleation reactions, example
        # {"name": {'k':f(n1, temp), 'i': 'SiO', 'mw': 44}}
        self.nuc_reacs = {}
        # dictionary of accreation reactions, example
        # {"name": {'k':f(temp, ncl,), 'i': ['SiO'], 'o': ['SiO[s]']}}
        self.acc_reacs = {}
        # index lists for various purposes
        self.idl_clmat = []  # cloud particle materials
        self.idl_vsed = [0]  # species that gravitationally settle

        # ==== Misc settings
        self.rg_fit_deg = 8  # degree of the polynomial to fit the ittarative radius

        # ==== storage variables
        self.results = {}  # stores results
        self.rg_history = None  # stores history of rg.
        self.loop_nr = 0  # remember the current loop number
        self.itterations = 0  # remember number of itterations performed

        # ==== Fudge factors to play around with simulation
        # IMPORTANT: The default values here correspond to a non-fudged run
        self.nuc_rate_fudge = 1  # factor to reduce or increase nucleation rate
        self.acc_rate_fudge = 1  # factor to reduce or increase growth rate

        # ==== Welcom message
        print('===========================================================')
        print('                   Welcome to Nimbus                       ')
        print('===========================================================')
        print('[INFO] For questions contact: kiefersv.mail@gmail.com')
        print('[INFO] Settings selected:')
        print('       -> working directory: ' + self.working_dir[:-1])
        print('       -> verbose: ' + str(verbose))
        print('       -> analytic plots: ' + str(create_analytic_plots))

