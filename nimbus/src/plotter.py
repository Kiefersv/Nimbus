import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import virga.justdoit as jdi
import astropy.units as u

#   universal gas constant (erg/mol/K)
RGAS = 8.3143e7
AVOG = 6.02e23
KB = RGAS / AVOG
PF = 1000000  # Reference pressure [dyn / cm2]

def plot_full_structure(self, y, title=''):
    """
    Plot the cloud structure.

    :param self: Nimbus object
    :param y: solution of solve ivp
    :param title: title for the plot
    :return:
    """

    # ==== General plotting set up
    plt.figure()
    plt.title(title)
    plt.plot([], [], color='k', linestyle='-', label='xv')
    plt.plot([], [], color='k', linestyle=':', label='xc')
    plt.plot([], [], color='k', linestyle='--', label='xn')

    # ==== show time evolution and convergence of the model
    nit = len(y[0])-1
    for i in range(nit):
        plt.plot(y[:self.sz, i], self.pres, color=cm.viridis(i/nit), linestyle='-', alpha=0.3)
        plt.plot(y[self.sz:self.sz*2, i], self.pres, color=cm.viridis(i/nit), linestyle=':', alpha=0.3)
        plt.plot(y[self.sz*2:, i], self.pres, color=cm.viridis(i/nit), linestyle='--', alpha=0.3)

    # ==== Plot final time step
    plt.plot(y[:self.sz, -1], self.pres, color='k', linestyle='-')
    plt.plot(y[self.sz:self.sz*2, -1], self.pres, color='k', linestyle=':')
    plt.plot(y[self.sz*2:, -1], self.pres, color='k', linestyle='--')

    # ==== Plot cloud particle radius (initial and final)
    mp = np.nan_to_num((y[self.sz:self.sz*2, -1]+y[self.sz*2:, -1]) * self.m_ccn / y[self.sz*2:, -1])
    rg = np.cbrt(3*mp/(4*np.pi*self.rhop))
    rg = np.maximum(rg, self.r_ccn)
    plt.plot(self.rg_in[self.mask_psupsat], self.pres[self.mask_psupsat], color='k', linestyle='-.', label=r'r$_\mathrm{in}$ [$\mu$m]')
    plt.plot(self.rg[self.mask_psupsat], self.pres[self.mask_psupsat], color='orange', linestyle='-.', label=r'r$_\mathrm{new}$ [$\mu$m]')
    plt.plot(rg[self.mask_psupsat], self.pres[self.mask_psupsat], color='green', linestyle='-.', label=r'r$_\mathrm{out}$ [$\mu$m]')
    plt.vlines([1e-4], self.pres[-1], self.pres[0], linestyle='-.', color='gray', label='1 $\mu$m')
    #
    # ==== plot cloud particle number density
    ncl = y[self.sz * 2:, -1] / self.m_ccn * self.rhoatmo
    ncl_o_ngas = ncl / self.natmo
    plt.plot(ncl_o_ngas, self.pres, color='blue', label='n/ngas', linestyle='--')

    # ==== Nucleation and growth rate
    n1 = y[:self.sz, -1] * self.rhoatmo / self.m1
    nuc_rate = np.nan_to_num(self.nuc_rate(n1, self.temp))
    growth_rate = self.acc_rate(self.rg_in, self.temp, n1, ncl)
    plt.plot(nuc_rate, self.pres, label='nucleation rate', color='red', linestyle='-')
    plt.plot(growth_rate, self.pres, label='accretion rate', color='magenta', linestyle='-')
    #
    # ==== Plot vapour pressure limit
    plt.plot(self.pvap * self.mw / self.pres / self.mmw, self.pres, label='q_vap', color='blue', linestyle='-.')

    # ==== General plotting settings
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc=2)
    plt.xlim(1e-18, 1e0)
    plt.ylim(self.pres[-1], self.pres[0])
    plt.ylabel('pres [dyne/cm2]')
    plt.xlabel('various')
    plt.savefig(self.working_dir + '/ap_structure_' + title + '.png')
    plt.show()

def plot_initial_conditions(self, x0):
    """
    Simple plotting routine to analyse the initial conditions.

    :param self: Nimbus clss
    :param x0: initial mass mixing ratios
    """
    plt.figure()
    plt.loglog(x0[:self.sz], self.pres, label='xv')
    plt.loglog(x0[self.sz:2*self.sz], self.pres, label='xc')
    plt.loglog(x0[self.sz*2:3*self.sz], self.pres, label='xn')
    plt.ylim(self.pres[-1], self.pres[0])
    plt.legend()
    plt.ylabel('pres [dyne/cm2]')
    plt.xlabel('MMR [g/g]')
    plt.savefig(self.working_dir + '/ap_initial_profiles.png')

def plot_spectrum(self, yin, type='transmission'):
    qext, qscat, cos_qscat, nwave, radius, wave_in = jdi.get_mie('SiO',
                                                                 '/home/kiefersv/Documents/work/aa_clean/opacities/')
    qc = np.asarray([yin[self.sz:self.sz * 2]]).T
    ndz = np.asarray([-yin[self.sz * 2:self.sz * 3] / self.m_ccn * self.dz]).T
    rgin = np.asarray([self.rg]).T
    qext = qext[:, :, np.newaxis]
    qscat = qscat[:, :, np.newaxis]
    cos_qscat = cos_qscat[:, :, np.newaxis]
    nradii = len(radius)
    rmin = np.min(radius)
    rmax = np.max(radius)

    radius, rup, dr = jdi.get_r_grid_w_max(rmin, rmax, n_radii=nradii)
    opd, w0, g0, opd_gas = jdi.calc_optics(nwave, qc, qc, rgin, rgin, ndz, radius,
                                           dr, qext, qscat, cos_qscat, 2, rmin, nradii)

    print(opd)
    df_cloud = jdi.picaso_format(
        opd[:-1], w0[:-1], g0[:-1],
        pressure=10 ** self.logp_mid * 1e-6, wavenumber=1 / wave_in[:, 0] / 1e-4,
    )
    # ==== Planet data
    tcase = {}
    tcase['gravity'] = 10 ** 2.49  # (Zhang et al. 2024)
    tcase['M_planet'] = 0.12  # in jupiter mass (Wood et al. 2023, also Bohn et al. 2020ab)
    tcase['R_planet'] = 0.948  # in jupiter radii (Bohn et al. 2020ab)
    tcase['mh_planet'] = 1  # in solar metalicity (assumed)
    tcase['mmw'] = 2.34  # (assumed)

    # ==== Stellar data
    tcase['R_star'] = 0.66  # in solar radii (Dai et al. 2018)
    tcase['T_star'] = 4430  # in K (Dai et al. 2018)
    tcase['mh_star'] = 0.02  # in solar metalicity (Dai et al. 2018)
    tcase['logg_star'] = 4.5  # in solar metalicity (Dai et al. 2018)
    tcase['distance'] = 64.7  # in pc (from dyrek et al. 2024)

    import picaso.justdoit as jdip
    opa = jdip.opannection(wave_range=[3, 15])
    case1 = jdip.inputs()
    case1.phase_angle(0)
    case1.gravity(mass=tcase['M_planet'], mass_unit=u.Unit('M_jup'),
                  radius=tcase['R_planet'], radius_unit=u.Unit('R_jup'))
    case1.star(opa, tcase['T_star'], tcase['mh_star'], tcase['logg_star'],
               radius=tcase['R_star'], radius_unit=jdip.u.Unit('R_sun'))

    # chemistry from ARCiS fit
    tcase['add_gas_phase_species_constant'] = {
        'H2': 1,
        'H2O': 10 ** -2.19,
        'SO2': 10 ** -5.03,  # not available
        'H2S': 10 ** -2.65,
        'NH3': 10 ** -5.47,
        'CO': 10 ** -2.41,
        'PH3': 10 ** -6.29,  # insignificant
        'HCN': 10 ** -9.26,  # insignificant, not available
        'C2H2': 10 ** -9.08,  # insignificant, not available
        'SiO': 10 ** -6.08,  # insignificant, not available
        'CH4': 10 ** -8.52,  # insignificant
        'CO2': 10 ** -8.05,  # insignificant
        'SO': 10 ** -7.38,  # insignificant, not available
    }
    chem = {}
    oa = np.ones_like(self.pres)
    if 'add_gas_phase_species_constant' in tcase:
        for chem_spec in tcase['add_gas_phase_species_constant']:
            chem[chem_spec] = oa * tcase['add_gas_phase_species_constant'][chem_spec]
    tcase['chemistry'] = chem

    # ==== Get atmosphere from tcase
    d = {
        'pressure': self.pres * 1e-6,
        'temperature': self.temp,
    }
    for key in tcase['chemistry']:
        d[key] = tcase['chemistry'][key]
    df = pd.DataFrame(data=d)
    case1.atmosphere(df=df)
    case1.clouds(df=df_cloud)

    # ==== Calculate transmission and emission spectra
    t_df = case1.spectrum(opa, full_output=True, calculation='transmission')

    # import picaso.justplotit as jpip
    # full_output = t_df['full_output']
    # fig, ax, um, CF_bin = jpip.transmission_contribution(full_output, R=100)
    # fig.show()

    # ==== Regrid output
    trans_wavn, trans_rprs2 = t_df['wavenumber'], t_df['transit_depth']
    trans_wavn_bins, trans_wavn_rprs2 = jdip.mean_regrid(trans_wavn, trans_rprs2, R=150)

    plt.figure()
    plt.plot(1 / trans_wavn_bins * 1e4, trans_wavn_rprs2)
    plt.xscale('log')
    plt.savefig(self.working_dir + '/transmission_spectrum.png')