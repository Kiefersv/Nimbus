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
    from .src.settings import set_cloud_settings, set_solver_settings

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

        # ==== storage variables
        self.results = {}





