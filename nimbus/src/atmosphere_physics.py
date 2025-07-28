"""
Functions concerning atmospheric properties. Different functions might be used
or added.
"""

import numpy as np

def get_settling_velocity_function(self):
    """
    Function to prepare settling velocity functions.
    :param self: Nimbus class object
    """

    # ===================================================================================
    #  Settling velocity
    # ===================================================================================
    # Note: all settling velocity functions must be of the form f()
    """
    Settling velocity of cloud particles

    :param rg: cloud particle size [cm]
    :param temp: temperature [K]
    :return: terminal cloud particle settling velocity [cm/s] 
    """

    def _vsed_exolyn(rg, rhop, temp):
        """
        Settling velocity taken from ExoLyn (Huang et al. 2024):
        Citation:  	https://doi.org/10.1051/0004-6361/202451112
        Link: https://github.com/helonghuangastro/exolyn
        """
        # ==== Physical parameters
        vth = np.sqrt(self.rgas * temp / (2 * np.pi * self.mmw))  # thermal velocity [cm/s]
        # ==== settling velocity
        vsed = (self.gravity * rg * rhop / (vth * self.rhoatmo) *
                np.sqrt(1 + (4 * rg / (9 * self.lmfp)) ** 2))
        return vsed

    def _vsed_diffudrift(rg):
        """
        Settling velocity from DiffuDrift (Woitke et al. 2020):
        Citation: https://doi.org/10.1051/0004-6361/201936281
        """
        vsed = (np.sqrt(np.pi) * self.gravity * self.rhop *
                self.rg / 2 / self.rhoatmo / self.ct)
        return vsed

    def _vsed_virga(rg):
        """
        Settling velocity of VIRGA:
        Link: https://github.com/natashabatalha/virga
        """

        # ==== pyhsical properties
        knudsen = self.lmfp / rg  # knudesn number
        drho = self.rhop - self.rhoatmo  # difference in density
        # Cunningham correction (slip factor for gas kinetic effects)
        beta_slip = 1. + 1.26*knudsen
        # atmospheric viscosity (dyne s/cm^2) from VIRGA
        visc = (5./16. * np.sqrt(np.pi * self.kb * self.temp * (self.mmw / self.avog)) /
                self.cs_mol / (1.22 * (self.temp / self.eps_k) ** (-0.16)))

        # ==== Stokes terminal velocity (low Reynolds number)
        vfall_r = beta_slip*(2.0/9.0)*drho*self.gravity*rg**2 / visc
        # compute reynolds number for low reynolds number case
        reynolds = 2.0*rg*self.rhoatmo*vfall_r / visc

        # ==== High raynold number cases
        # if reynolds number is between 1-1000 we are in turbulent flow
        cd_nre2 = 32.0 * rg**3.0 * drho * self.rhoatmo * self.gravity / (3.0 * visc ** 2)
        # coefficients from EQN 10-111 in Pruppachar & Klett 1978
        # they are an empirical fit to Figure 10-9
        xx = np.log(cd_nre2)
        b0, b1, b2, b3 = -0.318657e1, 0.992696, -.153193e-2, -.987059e-3
        b4, b5, b6 = -.578878e-3, 0.855176e-4, -0.327815e-5
        y = b0 + b1*xx**1 + b2*xx**2 + b3*xx**3 + b4*xx**4 + b5*xx**5 + b6*xx**6
        vfall_r[reynolds > 1] = (visc*reynolds / (2.*rg*self.rhoatmo))[reynolds > 1]
        # if raynolds number > 1000
        cdrag = 0.45
        sqrtfac = np.sqrt(8. * drho * rg * self.gravity / (3. * cdrag * self.rhoatmo))
        vfall_r[reynolds > 1e3] = (beta_slip * sqrtfac)[reynolds > 1e3]

        return vfall_r

    return _vsed_exolyn

# =======================================================================================
#  Nucleation rates
# =======================================================================================
def get_nucleation_rate_function(self, r1, sig, pvap, mw):
    """
    Function to prepare nucleation rate functions.

    :param self: Nimbus object
    :param r1: float, radius of the nucleating monomer
    :param sig: function, surface tension of the nucleating species
    :param pvap: function, vapor pressure of the nucleating species
    """

    # Note: all nucleation rate functions must be of the form f(n1, temp) and have the
    # follwing header:
    """
    :param n1: number density of cloud forming material [1/cm3]
    :param temp: temperature [K]
    :return:
    """

    def _nuc_rate_mini_cloud(n1, temp):
        """
        This nucleation rate was taken from Elsie Lee's mini cloud:
        Citation: https://academic.oup.com/mnras/article/524/2/2918/7221353
        Link: https://github.com/ELeeAstro/mini_cloud
        """

        # ==== Hard coded values
        alpha = 1.0  # sticking coefficient []
        Nf = 5.0  # MCNT factor []

        # ==== Physical parameters
        p1 = n1 * self.kb * temp  # partial pressure [dyne/cm2]
        sat = p1 / pvap(temp)  # log of saturation []
        ln_ss = np.log(sat)  # log of supersaturation []
        f0 = 4.0 * np.pi * r1 ** 2  # colisional corsssection [cm2]
        kbT = self.kb * temp  # shorthand notation
        theta_inf = (f0 * sig(temp)) / kbT  # theta inf [?]

        # ==== Prevent unphysical sat values (will be removed at the end)
        ln_ss[ln_ss <= 1e-30] = 1e-30

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
                  * np.sqrt(kbT / (2.0 * np.pi * mw / self.avog)) * n1)

        # ==== everything together gives the nucleaiton rate
        exponent = np.maximum(-300.0, N_star_1 * ln_ss - dg_rt)
        f_nuc_hom = n1 * tau_gr * Zel * np.exp(exponent)

        # ==== Remove nans and other problems
        # Note: We only check here the legality of the saturation input to
        # allow for a vecotrised input
        f_nuc_hom[sat <= 1] = 0

        # ==== fudge with nucleation rate (No fudge: self.nuc_rate_fudge = 1)
        f_nuc_hom *= self.nuc_rate_fudge

        return f_nuc_hom

    def _nuc_rate_sindel(n1, temp):
        """
        This nucleation rate follows the description of Sindel et al. (2022):
        Citation:  https://doi.org/10.1051/0004-6361/202243306
        """

        # ==== Hard coded
        alpha = 1.0  # sticking coefficient []

        # ==== Physical parameters
        p1 = n1 * self.kb * temp  # partial pressure [dyne/cm2]
        sat = p1 / pvap(temp)  # log of saturation []
        ln_ss = np.log(sat)  # log of supersaturation []
        f0 = 4.0 * np.pi * r1 ** 2  # colisional corsssection [cm2]
        kbT = self.kb * temp  # shorthand notation
        theta_inf = (f0 * sig(temp)) / kbT  # theta inf [?]

        # ==== Prevent unphysical sat values (will be removed at the end)
        ln_ss[ln_ss <= 1e-30] = 1e-30

        # ==== Calcualte cirtical cluster size
        N_inf = (((2.0 / 3.0) * theta_inf) / ln_ss) ** 3
        N_star = 1.0 + (N_inf/8.0)
        N_star = np.maximum(1.00001, N_star)  # make sure Nstar-1 is not below 0
        N_star_1 = N_star - 1.0  # shorthand notation

        # ==== Gibbs free energy approximation
        dg_rt = theta_inf * N_star_1**(2/3)

        # ==== Zeldovich factor
        Zel = np.sqrt(theta_inf / (9.0 * np.pi * N_star_1**(4.0/3.0)))

        # ==== growth rate
        tau_gr = ((f0 * N_star**(2.0/3.0)) * alpha
                  * np.sqrt(kbT / (2.0 * np.pi * mw / self.avog)) * n1)

        # ==== everything together gives the nucleaiton rate
        exponent = np.maximum(-300.0, N_star_1 * ln_ss - dg_rt)
        f_nuc_hom = n1 * tau_gr * Zel * np.exp(exponent)

        # ==== Remove nans and other problems
        # Note: We only check here the legality of the saturation input to
        # allow for a vecotrised input
        f_nuc_hom[sat <= 1] = 0

        # ==== fudge with nucleation rate (No fudge: self.nuc_rate_fudge = 1)
        f_nuc_hom *= self.nuc_rate_fudge

        return f_nuc_hom

    # return the nucleation rate function for later use (all parameters fixed)
    return _nuc_rate_mini_cloud

# =======================================================================================
#  Accretion rates
# =======================================================================================
def get_accretion_rate_prefactor(self, r1, pvap, mw):

    # Note: all nucleation rate functions must be of the form f(rg, temp) and
    # have the follwing header:
    """
    :param rg: cloud particle size [cm]
    :param temp: temperature [K]

    :return: accretion rate [1/cm3]
    """

    def _acc_rate_mini_cloud(rg, temp):
        """
        Accretion rate following Lee (2023):
        Citation: https://doi.org/10.1093/mnras/stad2037
        Link: https://github.com/ELeeAstro/mini_cloud
        """

        # ==== Physical parameters
        vth = np.sqrt(self.rgas * temp / (2 * np.pi * mw))  # thermal velocity [cm/s]

        # ==== Gaseous diffusion constant
        d0 = 2*r1
        diff_const = (5.0/(16.0 * self.avog * d0**2 * self.rhoatmo) *
                      np.sqrt((self.rgas * temp * self.mmw)/(2.0 * np.pi) *
                              (mw + self.mmw)/mw))

        # ==== Accreation rate in two limits
        # high knudsen number limit
        dmdt_high = 4*np.pi * rg**2 * vth
        # low knudsen number limit
        dmdt_low = 4*np.pi * rg * diff_const
        # interpolate
        val_low = np.maximum(dmdt_low, 1e-30)
        val_high = np.maximum(dmdt_high, 1e-30)
        # fx = 0.5 * (1.0 - np.tanh(2.0*np.log10(val_low/val_high)))
        # dmdt = dmdt_low * fx + dmdt_high * (1.0 - fx)
        dmdt_pref = 1/(1/val_low + 1/val_high)  # changed interpolation scheme

        # ==== fudge with accretion rate (No fudge: self.nuc_rate_fudge = 1)
        dmdt_pref *= self.acc_rate_fudge

        return dmdt_pref

    def _acc_rate_sw(rg, temp):
        """
        Accretion rate following Helling et al. (2006) assuming collisional regim:
        Citation: https://www.aanda.org/10.1051/0004-6361:20054598
        """

        # ==== growth rate
        growth_rate_pref = 4*np.pi * rg**2 * self.vth

        # ==== fudge with accretion rate (No fudge: self.nuc_rate_fudge = 1)
        growth_rate_pref *= self.acc_rate_fudge

        return growth_rate_pref

    # return the accretion rate function for later use (all parameters fixed)
    return _acc_rate_mini_cloud

# =======================================================================================
#  Fixed functions that don't need to be changed
# =======================================================================================

def mass_to_radius(self, xn, xc, rhop):
    """
    Calculate cloud particle radius from mass

    Parameters
    ----------
    xn : np.ndarray
        Cloud particle number density in mass mixing ratio [g/g]
    xc : np.ndarray
        Cloud particle mass mixing ratio [g/g]

    Return
    ------
    radius : np.ndarray
        Cloud particle radius [cm]
    """
    mp = np.nan_to_num(xc * self.m_ccn / xn)  # cloud particle mass [g]
    rg = np.cbrt(3 * mp / (4 * np.pi * rhop)) # cloud particle radius [cm]
    rg = np.maximum(rg, self.r_ccn)  # prevent low values
    return rg

def get_rhop(self, x):
    """
    Calcualte density of mixed cloud particle.

    Parameters
    ----------
    x : np.ndarray
        Cloud particle mass mixing ratio [g/g]
    """
    vtot_p_rhoatmo = np.sum(x[self.idl_clmat] / self.clmat_rhop)
    mtot_p_rhoatmo = np.sum(x[self.idl_clmat])
    return mtot_p_rhoatmo / vtot_p_rhoatmo


# # ===================================================================================
# #  Coagoulation rate
# # ===================================================================================
# # Not implemented yet
# def _coag_mini_cloud(rg):
#
#     # atmospheric viscosity (dyne s/cm^2) from VIRGA
#     # EQN B2 in A & M 2001, originally from Rosner+2000
#     # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
#     visc = (5. / 16. * np.sqrt(np.pi * self.kb * self.temp * (self.mmw / self.avog)) /
#             self.cs_mol / (1.22 * (self.temp / self.ps_k) ** (-0.16)))
#
#     # Knudsen number
#     Kn = self.lmfp/rg
#
#     # cloud particle mass
#     m_c = np.maximum(4/3 * np.pi *rg**3 * self.rhop, self.m_ccn)
#
#     # Cunningham slip factor (Kim et al. 2005)
#     Kn_b = min(Kn, 100.0)
#     beta = 1.0 + Kn_b*(1.165 + 0.483 * np.exp(-0.997/Kn_b))
#
#     # Particle diffusion rate
#     D_r = (self.kb*self.temp*beta)/(6.0*np.pi*visc*rg)
#
#     # Thermal velocity limit rate
#     V_r = np.sqrt((8.0*self.kb*self.temp)/(np.pi*m_c))
#
#     # Moran (2022) method using diffusive Knudsen number
#     Knd = (8.0*D_r)/(np.pi*V_r*rg)
#     phi = 1.0/np.sqrt(1.0 + np.pi**2/8.0 * Knd**2)
#     f_coag = (-4.0*self.kb*self.temp*beta)/(3.0*visc) * phi
#
#     return f_coag

# ===================================================================================
#  Set functions
# ===================================================================================