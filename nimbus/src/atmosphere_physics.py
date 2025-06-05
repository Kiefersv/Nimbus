import numpy as np

#   universal gas constant (erg/mol/K)
RGAS = 8.3143e7
AVOG = 6.02e23
KB = RGAS / AVOG
PF = 1000000  # Reference pressure [dyn / cm2]

def define_atmosphere_physics(self):
    """
    Set up all functions that handle the microphysics of cloud formation. This includes:
        - Nucleation rate (Available: MCNT)
        - Growth rate (Available: SW)
    :param self: Nimbus class object
    """

    # ===================================================================================
    #  Nucleation rates
    # ===================================================================================
    # Note: all nucleation rate functions must be of the form f(n1, temp)

    def _nuc_rate_mini_cloud(n1, temp):
        """
        This nucleation rate was taken from Elsie Lee's mini cloud:

        Citation: https://academic.oup.com/mnras/article/524/2/2918/7221353
        Link: https://github.com/ELeeAstro/mini_cloud

        :param n1: number density of cloud forming material [1/cm3]
        :param temp: temperature [K]
        :return:
        """

        # ==== Hard coded
        alpha = 1.0  # sticking coefficient []
        Nf = 5.0  # MCNT factor []

        # ==== Physical parameters
        p1 = n1 * KB * temp  # partial pressure [dyne/cm2]
        sat = p1 / self.pvap  # log of saturation []
        ln_ss = np.log(sat)  # log of supersaturation []
        f0 = 4.0 * np.pi * self.r1 ** 2  # colisional corsssection [cm2]
        kbT = KB * temp  # shorthand notation
        theta_inf = (f0 * self.sig) / kbT  # theta inf [?]

        # ==== Calcualte cirtical cluster size
        N_inf = (((2.0 / 3.0) * theta_inf) / ln_ss) ** 3
        N_star = 1.0 + (N_inf/8.0) * (1.0 + np.sqrt(1.0 + 2.0 * (Nf/N_inf)**(1/3))
                                      -2.0 * (Nf/N_inf)**(1/3))**3
        N_star = np.maximum(1.00001, N_star)  # make sure Nstar-1 is not below 0
        N_star_1 = N_star - 1.0  # shorthand notation

        # ==== Gibbs free energy approximation
        dg_rt = theta_inf * (N_star_1 / (N_star_1**(1/3) + Nf**(1/3)))

        # ==== Zeldovich factor
        Zel = np.sqrt((theta_inf / (9.0 * np.pi * N_star_1**(4.0/3.0)))
                      * ((1.0 + 2.0 * (Nf/N_star_1)**(1/3))
                      / (1.0 + (Nf/N_star_1)**(1/3))**3))

        # ==== growth rate
        tau_gr = ((f0 * N_star**(2.0/3.0)) * alpha
                  * np.sqrt(kbT / (2.0 * np.pi * self.mw / AVOG)) * n1)

        # ==== everything together gives the nucleaiton rate
        f_nuc_hom = n1 * tau_gr * Zel * np.exp(np.maximum(-300.0, N_star_1 * ln_ss - dg_rt))

        # ==== Remove nans and other problems
        # Note: We only check here the legality of the saturation input to
        # allow for a vecotrised input
        f_nuc_hom[sat <= 1] = 0

        # ==== fudge with nucleation rate (No fudge: self.nuc_rate_fudge = 1)
        f_nuc_hom *= self.nuc_rate_fudge

        return f_nuc_hom

    # ===================================================================================
    #  Accretion rates
    # ===================================================================================
    # Note: all nucleation rate functions must be of the form f(rg, temp, n1, ncl)

    def _acc_rate_mini_cloud(rg, temp, n1, ncl):
        """
        Accretion rate following Helling et al. 2006 assuming collisional regim:

        Citation: https://www.aanda.org/10.1051/0004-6361:20054598

        :param rg: cloud particle size [cm]
        :param temp: temperature [K]
        :param n1: number density of cloud forming material [1/cm3]
        :param ncl: cloud particle number density [1/cm3]

        :return: accretion rate [1/cm3]
        """

        # Physical parameters
        p1 = n1 * KB * temp  # partial pressure [dyne/cm2]
        kn = self.lmfp/rg # knudsen number
        # Gaseous diffusion constant
        diff_const = 5.0/(16.0*AVOG*2*rg**2*self.rhoatmo) * np.sqrt((RGAS*self.temp*self.mmw)/(2.0*np.pi) * (self.mw + self.mmw)/self.mw)

        # high knudsen number limit
        dmdt_high = 4*np.pi * rg**2 * n1 * ncl * self.vth * (1 - self.pvap/p1)
        # low knudsen number limit
        dmdt_low = 4*np.pi * rg * diff_const * self.vth * (1 - self.pvap/p1)


        # interpolate
        kn_crit = 1/3 # kn * (dmdt_high/dmdt_low)
        knd = kn/kn_crit
        fx = 0.5 * (1.0 - np.tanh(2.0*np.log10(knd)))
        dmdt = dmdt_low * fx + dmdt_high * (1.0 - fx)

        # ==== fudge with accretion rate (No fudge: self.nuc_rate_fudge = 1)
        dmdt *= self.acc_rate_fudge

        return dmdt



    def _acc_rate_sw(rg, temp, n1, ncl):
        """
        Accretion rate following Helling et al. 2006 assuming collisional regim:

        Citation: https://www.aanda.org/10.1051/0004-6361:20054598

        :param rg: cloud particle size [cm]
        :param temp: temperature [K]
        :param n1: number density of cloud forming material [1/cm3]
        :param ncl: cloud particle number density [1/cm3]

        :return:
        """

        # ==== Physical parameters
        p1 = n1 * KB * temp  # partial pressure [dyne/cm2]

        # ==== growth rate
        growth_rate = 4 * np.pi * rg ** 2  * n1 * ncl * self.vth * (1 - self.pvap / p1)

        # ==== fudge with accretion rate (No fudge: self.nuc_rate_fudge = 1)
        growth_rate *= self.acc_rate_fudge

        return growth_rate

    # ===================================================================================
    #  Settling velocity
    # ===================================================================================
    # Note: all settling velocity functions must be of the form f()

    def _vsed_functions_exolyn(rg):
        """
        Settling velocity of cloud particles

        :param rg: cloud particle size [cm]
        :return:
        """
        vsed = (self.gravity * rg * self.rhop / (self.vth * self.rhoatmo) *
                np.sqrt(1 + (4 * rg / (9 * self.lmfp)) ** 2))
        return vsed



    # ===================================================================================
    #  Coagoulation rate
    # ===================================================================================
    def _coag_mini_cloud(rg):

        # atmospheric viscosity (dyne s/cm^2) from VIRGA
        # EQN B2 in A & M 2001, originally from Rosner+2000
        # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
        visc = (5. / 16. * np.sqrt(np.pi * KB * self.temp * (self.mmw / AVOG)) /
                self.cs_mol / (1.22 * (self.temp / self.ps_k) ** (-0.16)))

        # Knudsen number
        Kn = self.lmfp/rg

        # cloud particle mass
        m_c = np.maximum(4/3 * np.pi *rg**3 * self.rhop, self.m_ccn)

        # Cunningham slip factor (Kim et al. 2005)
        Kn_b = min(Kn, 100.0)
        beta = 1.0 + Kn_b*(1.165 + 0.483 * np.exp(-0.997/Kn_b))

        # Particle diffusion rate
        D_r = (KB*self.temp*beta)/(6.0*np.pi*visc*rg)

        # Thermal velocity limit rate
        V_r = np.sqrt((8.0*KB*self.temp)/(np.pi*m_c))

        # Moran (2022) method using diffusive Knudsen number
        Knd = (8.0*D_r)/(np.pi*V_r*rg)
        phi = 1.0/np.sqrt(1.0 + np.pi**2/8.0 * Knd**2)
        f_coag = (-4.0*KB*self.temp*beta)/(3.0*visc) * phi

        return f_coag

    # ===================================================================================
    #  Set functions
    # ===================================================================================
    self.nuc_rate = _nuc_rate_mini_cloud
    self.acc_rate = _acc_rate_sw
    self.vsed = _vsed_functions_exolyn