
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import xarray as xr

from pvap import _get_pvap
from solver import set_up_fex
from atmosphere_physics import set_up_atmosphere_physics
from plotter import plot_initial_conditions, plot_full_structure, plot_spectrum
from species_database import DataStorage

#   universal gas constant (erg/mol/K)
RGAS = 8.3143e7
AVOG = 6.02e23
KB = RGAS / AVOG
PF = 1000000  # Reference pressure [dyn / cm2]

class Nimbus():
    def __init__(self, temperature, pressure, kzz, mmw, gravity, fsed, specie, deep_mmr, working_dir='.',
                 create_analytic_plots=False):

        # ==== Workflow settings
        self.working_dir = working_dir  # working directory
        self.do_plots = create_analytic_plots  # If true, analytic plots are created
    
        # ==== Setting input parameters
        self.temp = temperature  # temperature profile [K]
        self.pres = pressure  # pressure profile [dyn/cm2]
        self.kzz = kzz  # mixing coefficient [cm2/s]
        self.mmw = mmw  # mean molecular weight [amu]
        self.gravity = gravity  # gravity [cm/s2]
        self.fsed = fsed  # (initial) settling parameter [None]
        self.deep_gas_mmr = deep_mmr  # mass mixing ratio in the interior [g/g]

        # ==== Set nucleation rate, accretion rate, and settling velocity
        set_up_atmosphere_physics(self)

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
        self.logp_mid = (self.logp[1:] + self.logp[:-1])/2  # midpoints
        # pressure grid bin size
        self.dlogp = np.zeros_like(self.logp)
        self.dlogp[1:-1] = (self.logp[2:] - self.logp[:-2])/2
        self.dlogp[0] = self.logp[1] - self.logp[0]
        self.dlogp[-1] = self.logp[-1] - self.logp[-2]
        # midpoints bin size
        self.dlogp_mid = self.logp[1:] - self.logp[:-1]
        # derivatives to be used later
        self.dz = - RGAS * self.temp / self.mmw / self.gravity * self.dlogp

        # ==== Derive physical properties
        self.natmo = self.pres / self.temp / KB  # total gas-phase number density [1/cm3]
        self.rhoatmo = self.mmw * self.pres / self.temp / RGAS  # atmospheric density [g/cm]
        self.m_ccn = 4/3 * np.pi * self.r_ccn**3 * self.rho_ccn  # ccn mass [g]
        self.vth = np.sqrt(RGAS*self.temp/(2*np.pi*self.mw))  # thermal velocity [cm/s]
        lmfpfac = np.sqrt(2) * self.rhoatmo * self.cs_mol
        self.lmfp = self.mmw / AVOG / lmfpfac  # mean free path [cm]
        self.h = RGAS * self.temp / self.gravity / self.mmw  # scale height [cm]
        self.ct = np.sqrt(2 * RGAS * self.temp / self.mmw)  # sound speed [cm/s]

        # ==== mid point values
        self.kzz_mid = np.interp(self.logp_mid, self.logp, self.kzz)
        self.rhoatmo_mid = np.interp(self.logp_mid, self.logp, self.rhoatmo)
        self.temp_mid = np.interp(self.logp_mid, self.logp, self.temp)
        self.dz_mid = - RGAS * self.temp_mid / self.mmw / gravity * self.dlogp_mid

        # ==== find pressures which are supersaturated
        ndeep = self.deep_gas_mmr * self.rhoatmo / self.m1
        pdeep = ndeep * KB * self.temp
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
            self.rg[i] = np.maximum(root_scalar(vsed_f, x0=self.r1*1e2).root, self.r_ccn)


    def compute(self, output_stepping=10, save_file=None):

        # ==== initial conditions
        x0 = self.set_initial_condidtions()  # load initial conditions
        # plot initial conditions
        if self.do_plots:
            plot_initial_conditions(self, x0)

        # ==== Variables to save intermediate results
        self.rg_history = np.zeros((len(self.pres), output_stepping+2))

        # ==== Main computational loop ==================================================
        # This loop iterates of cloud particle size. In each loop, rg is held constant
        # and updated at the end of the loop.
        yin = x0  # set initial condition for first loop
        for t, tit in enumerate(np.linspace(0, output_stepping, output_stepping+1)):
            self.rg_in = self.rg
            ts = np.logspace(np.log10(self.tstart), np.log10(self.tend), self.tsteps)
            sol = solve_ivp(
                self.fex, [self.tstart, self.tend], yin, method='LSODA',
                rtol=self.ode_rtol, atol=self.ode_atol, t_eval=ts#, jac=self.jac  !! jacobin does not lead to improvment
            )
            print(sol)
            yin = sol.y[:, -1]

            # ==== update cloud particle radius
            mp = np.nan_to_num(sol.y[self.sz:self.sz*2, -1] * self.m_ccn / sol.y[self.sz*2:, -1])
            rg = np.cbrt(3*mp/(4*np.pi*self.rhop))
            rg = np.maximum(rg, self.r_ccn)
            # create a polnom fit to the cloud particle radius to prevent sudden changes
            fit = np.polyval(np.polyfit(np.log10(self.pres[self.mask_psupsat]), np.log10(rg[self.mask_psupsat]), deg=3), np.log10(self.pres))
            rg = 10**fit
            rg = 10**((np.log10(rg) + np.log10(self.rg))/2)
            self.rg = np.maximum(rg, self.r_ccn)
            self.rg_history[:, t+1] = self.rg

            #plot_full_structure(self, sol.y, str(tit))

        # save data if a save file is given
        if not isinstance(save_file, type(None)):
            ds = xr.Dataset(
                data_vars = {
                    'qv': (['pressure'], sol.y[:self.sz, -1]),
                    'qc': (['pressure'], sol.y[self.sz:self.sz*2, -1]),
                    'qn': (['pressure'], sol.y[self.sz*2:self.sz*3, -1]),
                    'rg': (['pressure'], self.rg),
                },
                coords = {
                    'pressure': self.pres*1e-6,
                },
                attrs = {
                    'mmw': self.mmw,
                    'y': sol.y[:, -1],
                },
            )
            ds.to_netcdf(save_file + '.nc')

        return sol.y

    def set_up_fex(self):

        # ==== Set up ODE solver function
        set_up_fex(self)

    def load_previous_run(self, file_name):
        return xr.open_dataset(file_name)

    def set_solver_settings(
            self, initial_time_for_solver=None, end_time_for_solver=None,
            evaluation_steps_for_solver=20):
        if not isinstance(initial_time_for_solver, type(None)):
            self.tstart = initial_time_for_solver
        if not isinstance(end_time_for_solver, type(None)):
            self.tend = end_time_for_solver
        if not isinstance(evaluation_steps_for_solver, type(None)):
            self.tsteps = evaluation_steps_for_solver

    def parameter_settings(
            self, minimum_cloud_particle_radius=None, molecular_cross_section=None,
    ):
        if not isinstance(minimum_cloud_particle_radius, type(None)):
            self.r_ccn = minimum_cloud_particle_radius
        if not isinstance(molecular_cross_section, type(None)):
            self.cs_mol = molecular_cross_section

    def set_initial_condidtions(self):
        #x0 = self.set_initial_condidtions()
        x0 = np.zeros(3*self.sz) + 1e-30
        x0[:self.sz] = self.pvap * self.mw / self.pres / self.mmw
        x0[self.sz-6:self.sz] = self.deep_gas_mmr
        x0[:self.sz] = self.deep_gas_mmr
        x0[self.sz:self.sz*2] = 1e-30
        x0[self.sz*2:self.sz*3] = 1e-30

        return x0



