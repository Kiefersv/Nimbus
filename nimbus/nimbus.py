""" This is the header file of Nimbus. To find functionalities check the src folder. """

class Nimbus:
    """ Main class to handle the cloud structure calculation """

    # ==== physical constants
    rgas = 8.3143e7  # universal gas constant [erg/mol/K]
    avog = 6.02e23  # Avogadro constant [mol]
    kb = rgas / avog  # boltzmann constant [erg/K]
    pf = 1000000  # Reference pressure [dyn/cm2]

    # ==== import set up functions
    from .src.atmosphere import set_up_atmosphere
    from .src.solver import set_up_solver

    # ==== import the main compuational function
    from .src.compute import compute

    # ==== import solver settings
    from .src.settings import set_cloud_settings, set_solver_settings, set_fudge_settings

    # ==== import data handling functions
    from .src.data_storage import load_previous_run

    def __init__(self, working_dir='.',
                 create_analytic_plots=False):

        # ==== Workflow settings
        self.working_dir = working_dir + '/'  # working directory
        self.do_plots = create_analytic_plots  # If true, analytic plots are created

        # ==== initialisation checks
        self.isset_atmosphere = False  # checks if an atmosphere was initialised
        self.isset_solver = False  # checks if solver is set up
        self.isset_transmission_spectrum = False  # checks if all info for ts are given
        self.isset_emission_spectrum = False  # checks if all info for es are given

        # ==== working variables
        self.fex = None  # right hand side of time evolution (solved with solve_ivp)
        self.jac = None  # Jacobian matrix
        self.x0 = None  # initial mass mixing ratios

        # ==== Default solver settings (can be changed with set_solver_settings())
        self.tstart = 1e-4  # start time of simulation [s]
        self.tend = 1e15  # end time of simulation [s]
        self.tsteps = 20  # number of intermediated evaluations (log-spaced)
        self.ode_rtol = 1e-3  # relative error of solve_ivp
        self.ode_atol = 1e-25  # absolute error of solve_ivp

        # ==== storage variables
        self.results = {}  # stores results
        self.rg_history = None  # stores history of rg.

        # ==== Fudge factors to play around with simulation
        # IMPORTANT: The default values here correspond to a non-fudged run
        self.nuc_rate_fudge = 1  # factor to reduce or increase nucleation rate
        self.acc_rate_fudge = 1  # factor to reduce or increase growth rate





