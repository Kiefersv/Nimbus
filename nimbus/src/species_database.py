"""
This file contains all functions to load and assign data specific to gas-phase and cloud
material species.
"""
import os
import csv
import numpy as np
import xarray as xr

class DataStorage:
    """
    Storage of physical properties. Information on the stored variables:
    -> Cloud particle material:
        -> Surface tension [erg cm^-2] parameters in the form:
           surface_tension_A + surface_tension_B * T
        -> solid_density: Density of the solid material [g/cm3]
        -> monomer_radius: Radius of a single gas-phase molecule [cm]
        -> molecular_weight: Mass of a single gas-phase molecule [g]
        -> Vapor pressure [dyn/cm2] parameters in the form:
           pvap_prefactor * pvap_base**(
               pvap_A/T**2 + pvap_B/T + pvap_C + pvap_D*T + pvap_E*T**2 + pvap_F*T**2
           )
    """

    # pysical constants
    rgas = 8.3143e7  # Gas constant [erg/mol/K]
    avog = 6.02e23  # avogadro constant [1/mol]
    kb = rgas / avog  # boltzmann constant [erg/K]
    pf = 1000000  # Reference pressure [dyn / cm2]

    def __init__(self, data_file=None):
        # ==== Read in of parametrised cloud properties and basic values ================
        # open the cloud material data file and read it
        if data_file is None:
            data_file = os.path.dirname(__file__) + '/../data/chem/cloud_material.csv'
        raw_data = np.array(list(csv.reader(open(data_file))))
        # initialise data dict
        self.cloud_material_data = {}
        # loop over all species to initialise
        for s, spec in enumerate(raw_data[:, 0]):
            # skip header
            if s < 1:
                continue
            # get all data
            self.cloud_material_data[spec] = {
                'data_complete': raw_data[s, 1],
                'surface_tension_A': raw_data[s, 2],
                'surface_tension_B': raw_data[s, 3],
                'solid_density': float(raw_data[s, 4]),
                'monomer_radius': float(raw_data[s, 5]),
                'molecular_weight': float(raw_data[s, 6]),
                'pvap_base': raw_data[s, 7],
                'pvap_prefactor': raw_data[s, 8],
                'pvap_A': raw_data[s, 9],
                'pvap_B': raw_data[s, 10],
                'pvap_C': raw_data[s, 11],
                'pvap_D': raw_data[s, 12],
                'pvap_E': raw_data[s, 13],
                'pvap_F': raw_data[s, 14],
            }

        # ==== Read in of Gibbs free energies ===============================================
        # kjpmol_to_ergpmol = 1e10
        # self.gibbs_janaf = xr.open_dataset(
        #     os.path.dirname(__file__) + '/../data/Gibbs/janaf.nc'
        # ) * kjpmol_to_ergpmol
        kjpmol_to_ergpmol = 1e10
        self.gibbs_janaf = xr.open_dataset(
            os.path.dirname(__file__) + '/../data/Gibbs/ggchem.nc'
        ) * kjpmol_to_ergpmol


    # =======================================================================================
    #   Simple physical properties calculation (derived not read in)
    # =======================================================================================
    def monomer_mass(self, species):
        r1 = self.cloud_material_data[species]['monomer_radius']
        rho = self.cloud_material_data[species]['solid_density']
        return 4/3 * np.pi * r1**3 * rho

    def specific_gas_constant(self, species):
        mw = self.cloud_material_data[species]['molecular_weight']
        return self.rgas / mw

    # =======================================================================================
    #   Simple getter functions of physical properties
    # =======================================================================================
    def monomer_radius(self, species):
        return self.cloud_material_data[species]['monomer_radius']

    def molecular_weight(self, species):
        return self.cloud_material_data[species]['molecular_weight']

    def solid_density(self, species):
        return self.cloud_material_data[species]['solid_density']

    def surface_tension(self, species, temp):
        a = self.cloud_material_data[species]['surface_tension_A']
        b = self.cloud_material_data[species]['surface_tension_B']
        # check if data is available
        if a == 'x' or b == 'x':
            raise ValueError("No surface tension available for " + species)
        return float(a) + float(b) * temp

    def gibbs_free_energy(self, species, temp):
        return self.gibbs_janaf[species].interp({"temp_" + species: temp}).values

    # =======================================================================================
    #   Complex physical properties calculation (derived not read in)
    # =======================================================================================
    def vapor_pressures(self, species, temp, metalicity=1):
        """
        Data according to Lee et al. 2018 (A&A 614, A126)

        :param species: Name of species, see below for supported
        :param temp: Temperature in Kelvin, can be float or array
        :param metalicity: metalicity, optional (only used by few pvaps)
        :return:
            pvap : vapor pressure
        """

        # short hand notation
        mh = metalicity

        # Check if temp is an array, make it one if not
        isarray = True
        if isinstance(temp, float) or isinstance(temp, int):
            temp = np.asarray([temp])
            isarray = False

        # if all data is available, just read it in
        if self.cloud_material_data[species]['data_complete'] == 'yes':

            # get base of vapour pressure exponent
            base = None
            if self.cloud_material_data[species]['pvap_base'] == '10':
                base = 10
            elif self.cloud_material_data[species]['pvap_base'] == 'exp':
                base = np.exp(1)
            # get factors
            pref = float(self.cloud_material_data[species]['pvap_prefactor'])
            a = float(self.cloud_material_data[species]['pvap_A'])
            b = float(self.cloud_material_data[species]['pvap_B'])
            c = float(self.cloud_material_data[species]['pvap_C'])
            d = float(self.cloud_material_data[species]['pvap_D'])
            e = float(self.cloud_material_data[species]['pvap_E'])
            f = float(self.cloud_material_data[species]['pvap_F'])

            # calculate vapor pressure using data from file
            pvap = pref*base**(a/temp**2 + b/temp + c + d*temp + e*temp**2 + f*temp**3)

        # if there is a special case, treat it uniquely below
        elif self.cloud_material_data[species]['data_complete'] == 'special':
            # many special cases have casese which require arryfication
            pvap = np.zeros_like(temp)

            if species == 'C':
                pvap[:] = np.exp(3.27860e1 - 8.65139e4 / (temp + 4.80395e-1))

            elif species == 'CH4':
                pvap[:] = 10 ** (3.9895 - 443.028 / (temp - 0.49)) * 1e6

            elif species == 'Fe':
                pvap[temp > 1800.0] = np.exp(9.86 - 37120.0 / temp[temp > 1800.0]) * 1e6
                pvap[temp < 1800.0] = np.exp(15.71 - 47664.0 / temp[temp < 1800.0]) * 1e6

            elif species == 'H2O':
                temp_c = temp - 273.16
                pvap[temp_c<0] = 6111.5 * np.exp((23.036*temp_c[temp_c<0] - temp_c[temp_c<0]**2/333.7) / (temp_c[temp_c<0] + 279.82))
                pvap[temp_c>=0] = 6112.1 * np.exp((18.729*temp_c[temp_c>=0] - temp_c[temp_c>=0]**2/227.3) / (temp_c[temp_c>=0] + 257.87))

            elif species == 'H2S': # Stull(1947)
                pvap[:] = 10.0 ** (4.52887 - 958.587 / (temp - 0.539)) * 1e6
                pvap[temp<212.8] = 10.0**(4.43681 - 829.439 / (temp[temp<212.8] - 25.412)) * 1e6
                pvap[temp<30] = 10.0**(4.43681 - 829.439 / (30.0 - 25.412)) * 1e6

            elif species == 'S2':
                pvap[:] = np.exp(16.1 - 14000.0 / temp) * 1e6
                pvap[temp<413.0] = np.exp(27.0 - 18500.0 / temp[temp<413.0]) * 1e6

            elif species == 'S8':
                pvap[:] = np.exp(9.6 - 7510.0 / temp) * 1e6
                pvap[temp<413.0] = np.exp(20.0 - 11800.0 / temp[temp<413.0]) * 1e6

            elif species == 'SiO2':
                # this vapor pressure includes a metalicity correction
                pvap = 10.0**(13.168 - 28265/temp)*1e6/mh  # SiO + H2O -> SiO2[s] + H2

            else:
                raise ValueError('The species "' + species + '" is flagged as special, but no '
                                                          'case handling is provided.')

        # if neither is the case, data is missing
        else:
            raise ValueError('Data for nucleating species "' + species + '" not complete.')

        # If temp was not an array, return float
        if not isarray:
            pvap = pvap[0]

        return pvap

    # def reaction_supersaturation(self, cloud_specie, gas_species_in,
    #                             gas_species_out, temp):
    #     """
    #     Calculate teh reaction supersaturation according to Kiefer et al. 2024a
    #
    #     :param cloud_specie: str, name of cloud specie (should end in [s])
    #     :param species_in: Dict including names and VMR of Reactants
    #         Example: {'H2O': 2e-2, 'H2O': 2e-2, 'SiO': 1e-3}
    #     :param temp: Dict including names and VMR of (excluding solid)
    #     :return:
    #         pvap : vapor pressure
    #     """
    #
    #     # ==== energy of formation
    #     e_form = -self.gibbs_free_energy(cloud_specie, temp)
    #     for ins in gas_species_in:
    #         e_form += self.gibbs_free_energy(ins, temp)
    #     for outs in gas_species_out:
    #         e_form -= self.gibbs_free_energy(outs, temp)
    #
    #     # ==== number density prefactor
    #     n_fac = 1
    #     for ins in gas_species_in:
    #         n_fac *= gas_species_in[ins]
    #     for outs in gas_species_out:
    #         n_fac /= gas_species_out[outs]
    #
    #     # ==== reference pressure factor
    #     pow = len(gas_species_out) - len(gas_species_in)
    #     p_fac = (self.pf / self.kb / temp)**pow
    #
    #     # ==== reaction super saturation
    #     return n_fac * p_fac * np.exp(e_form/self.rgas/temp)
