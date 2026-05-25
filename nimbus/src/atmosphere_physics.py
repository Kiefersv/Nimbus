"""
Functions concerning atmospheric properties. Different functions might be used
or added.
"""
# pylint: disable=R0914

import numpy as np

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
    # Note: all nucleation rate functions must be of the form f(n1, temp, s) and
    # have the follwing header:
    # """
    # :param n1: number density of cloud forming material [1/cm3]
    # :param temp: temperature [K]
    # :param s: species index
    #
    # :return: accretion rate [1/cm3]
    # """
    def _nuc_rate_mini_cloud(n1, temp, s):
        """
        This nucleation rate was taken from Elsie Lee's mini cloud:
        Citation: https://academic.oup.com/mnras/article/524/2/2918/7221353
        Link: https://github.com/ELeeAstro/mini_cloud

        :param n1: number density of cloud forming material [1/cm3]
        :param temp: temperature [K]
        :return:
        """

        # ==== check if species nucleats
        if self.species[s] in self.ian:
            return np.zeros((len(temp),))

        # ==== Physical parameters
        pvap = self.db.vapor_pressures(self.species[s], self.temp, self.mh)
        sig = self.db.surface_tension(self.species[s], self.temp)

        # ==== Hard coded values
        alpha = 1.0  # sticking coefficient []
        nf = 5.0  # MCNT factor []

        # ==== Physical parameters
        p1 = n1 * self.kb * temp  # partial pressure [dyne/cm2]
        sat = p1 / pvap  # log of saturation []
        ln_ss = np.log(sat)  # log of supersaturation []
        f0 = 4.0 * np.pi * self.r1 ** 2  # colisional corsssection [cm2]
        kbt = self.kb * temp  # shorthand notation
        theta_inf = (f0 * sig) / kbt  # theta inf [?]

        # ==== Prevent unphysical sat values (will be removed at the end)
        ln_ss[ln_ss <= 1e-30] = 1e-30

        # ==== Calcualte cirtical cluster size
        n_inf = (((2.0 / 3.0) * theta_inf) / ln_ss) ** 3
        n_star = 1.0 + (n_inf/8.0) * (1.0 + np.sqrt(1.0 + 2.0 * (nf/n_inf)**(1/3))
                                      -2.0 * (nf/n_inf)**(1/3))**3
        n_star = np.maximum(1.00001, n_star)  # make sure Nstar-1 is not below 0
        n_star_1 = n_star - 1.0  # shorthand notation

        # ==== Gibbs free energy approximation
        dg_rt = theta_inf * (n_star_1 / (n_star_1**(1/3) + nf**(1/3)))

        # ==== Zeldovich factor
        zel = np.sqrt((theta_inf / (9.0 * np.pi * n_star_1**(4.0/3.0)))
                      * ((1.0 + 2.0 * (nf/n_star_1)**(1/3))
                      / (1.0 + (nf/n_star_1)**(1/3))**3))

        # ==== growth rate
        tau_gr = ((f0 * n_star**(2.0/3.0)) * alpha
                  * np.sqrt(kbt / (2.0 * np.pi * self.mw[s] / self.avog)) * n1)

        # ==== everything together gives the nucleaiton rate
        exponent = np.maximum(-300.0, n_star_1 * ln_ss - dg_rt)
        f_nuc_hom = n1 * tau_gr * zel * np.exp(exponent)

        # ==== Remove nans and other problems
        # Note: We only check here the legality of the saturation input to
        # allow for a vecotrised input
        f_nuc_hom[sat <= 1] = 0

        # ==== fudge with nucleation rate (No fudge: self.nuc_rate_fudge = 1)
        f_nuc_hom *= self.nuc_rate_fudge

        f_nuc_hom = np.maximum(f_nuc_hom, 0)

        return f_nuc_hom

    # ===================================================================================
    #  Accretion rates
    # ===================================================================================
    # Note: all nucleation rate functions must be of the form f(rg, temp, n1, ncl) and
    # have the follwing header:
    # """
    # :param rg: cloud particle size [cm]
    # :param temp: temperature [K]
    # :param n1: number density of cloud forming material [1/cm3]
    # :param ncl: cloud particle number density [1/cm3]
    #
    # :return: accretion rate [1/cm3]
    # """

    def _acc_rate_mini_cloud(rg, temp, n1, ncl, s):
        """
        Accretion rate following Lee (2023):
        Citation: https://doi.org/10.1093/mnras/stad2037
        Link: https://github.com/ELeeAstro/mini_cloud
        """

        # ==== Physical parameters
        p1 = n1 * self.kb * temp  # partial pressure [dyne/cm2]
        pvap = self.db.vapor_pressures(self.species[s], self.temp, self.mh)
        rvv = np.sqrt(self.rgas * self.temp / (2 * np.pi * self.mw[s]))  # rel vel of vapour [cm/s]

        # ==== Gaseous diffusion constant
        d0 = 2*self.r1
        diff_const = (5.0/(16.0 * self.avog * d0**2 * self.rhoatmo) *
                      np.sqrt((self.rgas * self.temp * self.mmw)/(2.0 * np.pi) *
                              (self.mw[s] + self.mmw)/self.mw[s]))

        # ==== Accreation rate in two limits
        # high knudsen number limit
        dmdt_high = 4*np.pi * rg**2 * n1 * ncl * rvv * (1 - pvap/p1)
        dmdt_high *= self.sticking_coefficient
        # low knudsen number limit
        dmdt_low = 4*np.pi * rg * n1 * ncl * diff_const * (1 - pvap/p1)
        # interpolate
        val_low = np.maximum(dmdt_low, 1e-200)
        val_high = np.maximum(dmdt_high, 1e-200)
        fx = 0.5 * (1.0 - np.tanh(2.0*np.log10(val_low/val_high)))
        fx = np.maximum(np.minimum(fx, 1.0), 1e-200)
        fx = np.maximum(fx, 0)
        dmdt = val_low * fx + val_high * (1.0 - fx)
        dmdt = np.nan_to_num(dmdt)
        dmdt = np.maximum(dmdt, 0)

        return dmdt

    # ===================================================================================
    #  Evaporation rates
    # ===================================================================================
    # Note: all nucleation rate functions must be of the form f(rg, temp, n1, ncl) and
    # have the follwing header:
    # """
    # :param rg: cloud particle size [cm]
    # :param temp: temperature [K]
    # :param n1: number density of cloud forming material [1/cm3]
    # :param ncl: cloud particle number density [1/cm3]
    #
    # :return: accretion rate [1/cm3]
    # """

    def _evp_rate_simple(rg, temp, n1, ncl, s):
        """
        Accretion rate following Lee (2023):
        Citation: https://doi.org/10.1093/mnras/stad2037
        Link: https://github.com/ELeeAstro/mini_cloud
        """

        # ==== Physical parameters
        p1 = n1 * self.kb * temp  # partial pressure [dyne/cm2]
        pvap = self.db.vapor_pressures(self.species[s], self.temp, self.mh)
        rvv = np.sqrt(self.rgas * self.temp / (2 * np.pi * self.mw[s]))  # rel vel of vapour [cm/s]

        # ==== Gaseous diffusion constant
        d0 = 2 * self.r1
        diff_const = (5.0 / (16.0 * self.avog * d0 ** 2 * self.rhoatmo) *
                      np.sqrt((self.rgas * self.temp * self.mmw) / (2.0 * np.pi) *
                              (self.mw[s] + self.mmw) / self.mw[s]))

        # ==== Accreation rate in two limits
        # high knudsen number limit
        dmdt_high = 4 * np.pi * rg ** 2 * n1 * ncl * rvv * (1 - pvap / p1)
        dmdt_high *= self.sticking_coefficient
        # low knudsen number limit
        dmdt_low = 4 * np.pi * rg * n1 * ncl * diff_const * (1 - pvap / p1)
        # interpolate
        val_low = np.minimum(dmdt_low, -1e-200)
        val_high = np.minimum(dmdt_high, -1e-200)
        fx = 0.5 * (1.0 - np.tanh(2.0 * np.log10(val_low / val_high)))
        fx = np.maximum(np.minimum(fx, 1.0), 1e-200)
        fx = np.maximum(fx, 0)
        dmdt = val_low * fx + val_high * (1.0 - fx)
        dmdt = np.nan_to_num(dmdt)
        dmdt = np.minimum(dmdt, 0)

        return dmdt

    # ===================================================================================
    #  Settling velocity
    # ===================================================================================
    # Note: all settling velocity functions must be of the form f()
    # """
    # Settling velocity of cloud particles
    #
    # :param rg: cloud particle size [cm]
    # :return: terminal cloud particle settling velocity [cm/s]
    # """

    def _vsed_exolyn(rg, rhop):
        """
        Settling velocity taken from ExoLyn (Huang et al. 2024):
        Citation:  	https://doi.org/10.1051/0004-6361/202451112
        Link: https://github.com/helonghuangastro/exolyn
        """
        vsed = (self.gravity * rg * rhop / (self.vth * self.rhoatmo) *
                np.sqrt(1 + (4 * rg / (9 * self.lmfp)) ** 2))
        vsed = np.nan_to_num(vsed)
        # # prevent unrealisticly high values
        # vsed[vsed > 1e30] = 1e30
        return vsed

    # # ===================================================================================
    # #  Coagoulation rate
    # # ===================================================================================
    # # Not implemented yet
    # def _coag_mini_cloud(rg):
    #
    #     # atmospheric viscosity (dyne s/cm^2) from VIRGA
    #     # EQN B2 in A & M 2001, originally from Rosner+2000
    #     # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems
    #     # (Dover: Mineola)
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
    self.nuc_rate = _nuc_rate_mini_cloud  # nucleation rate
    self.acc_rate = _acc_rate_mini_cloud  # accreation rate
    self.evp_rate = _evp_rate_simple  # evaporaiton rates
    self.vsed = _vsed_exolyn  # terminal settling velocity


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
