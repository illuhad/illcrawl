/*
 * This file is part of illcrawl, a reconstruction engine for data from
 * the illustris simulation.
 *
 * Copyright (C) 2017  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef QUANTITIES_CL
#define QUANTITIES_CL

#include "interpolation.cl"
#include "tabulated_function.cl"

#define X_H 0.76f

/// Boltzmann constant in eV/K
#define k_B 8.61733e-5f
/// adiabatic index
#define gamma (5.f/3.f)
/// Rydberg energy in eV
#define Ryd 13.605f

/// \return the mean molecular weight in units of proton masses
scalar mean_molecular_weight(scalar electron_abundance)
{
  return 4.f/(1.f + 3*X_H + 4*X_H*electron_abundance) /* * m_p*/;
}

/// \return The temperature in K
/// \param internal_energy The internal energy as given by Illustris
/// in units of (km/s)^2
/// \param the (dimensionless) electron abundance
scalar get_temperature(scalar internal_energy, scalar electron_abundance)
{
  // mean molecular weight in units of proton masses.
  scalar mu = mean_molecular_weight(electron_abundance);

  // The (specific) internal energy is given as (km/s)^2, but we want
  // to work in keV/g. 1 km^2/s^2 = 6.2415091e18 keV/g.
  // Furthermore, mu is in units of a proton mass (1.6726218e-24 g).
  // Therefore, to work in units of keV, the expression must be
  // multiplied by 1.6726218e-24 g * 6.2415091e18 keV/g.
  // Together with the division by k_B (in keV/K), the overall conversion
  // factor is 121.1475502.
  return  (gamma - 1.f) * internal_energy * mu * 121.1475502f;
}

/// \return gaunt factor for a given energy
/// \param E photon energy in keV
/// \param T plasma temperature in K
/// \param Z atomic number
/// \param gaunt_factor_table The thermally-averaged gaunt factors
/// as given by van Hoof (2014)
scalar g_ff(scalar E, scalar T, scalar Z,
            __read_only image2d_t gaunt_factor_table)
{
  // See van Hoof (2014) for more explanations regarding
  // these quantities
  scalar gam2 = Z * Z * Ryd / (k_B * T);

  // The factor of 1000 converts k_B*T from eV to keV
  scalar u = 1.e3f * E / (k_B * T);

  return evaluate_tabulated_function2d(gaunt_factor_table,
                                       (vector2)(-6.f, -16.f), // log(gam2), log(u) values of first element
                                       (vector2)(0.2f, 0.2f), // spacing of log(gam2), log(u) entries in table
                                       (vector2)(log(gam2), log(u))); // evaluation position
}

/// \return The thermal Bremsstrahlung X-Ray flux in units of keV/s/kpc^3/m^2
///  for a hydrogen plasma within a given energy bin
/// \param E the photon energy in keV
/// \param dE the energy bin width in keV. if dE=1, the returned value
/// corresponds to the differential value with respect to the energy of the
/// emitted photons, i.e. in units keV/s/kpc^3/keV/m^2 (because dE is simply
/// multiplied with the result)
/// \param T the plasma temperature in K
/// \param electron_abundance the electron abundance (dimensionless)
/// \param density the density in units of M_sun/kpc^3
/// \param r the distance to the observer in kpc (luminosity distance)
/// \param gaunt_table tabulated gaunt factors
scalar get_spectral_brems_emission(scalar E,
                                   scalar dE,
                                   scalar T,
                                   scalar electron_abundance,
                                   scalar density,
                                   scalar r,
                                   __read_only image2d_t gaunt_table)
{
  // convert density to a number density
  // n = rho/m_p
  // n [1/kpc^3] = rho [M_sun/kpc^3] * 1.9884e33g / 1.672621898e-24g
  // n [1/kpc^3] = rho [M_sun/kpc^3] * 1.18879e57
  // n [1/cm^3] = rho [M_sun/kpc^3] * 4.046263e-08
  // Factor of 1e-8 will be directly absorbed into the prefactor
  // of the bremsstrahlung emission
  scalar ion_density = 4.046263f * density * electron_abundance;
  scalar Z = 1.f;

  // This is 1/(k_B*T) in keV.
  // The factor 1000 turns eV into keV.
  scalar T_keV_inv = 1000.f / (k_B * T);

  // Bremsstrahlung per frequency per volume per time is given by
  // Brems = 6.8415764e-38*1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) ergs*s^(-1)*cm^(-3)*Hz^-1
  //       = 4.2701761e-29 1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*cm^(-3)*Hz^-1
  // Brems_per_energy = Brems / h
  //                  = 64445.08 * 1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*cm^(-3)*J^(-1)
  //                  = 1.032524e-11 * 1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*cm^(-3)*keV^(-1)
  //                  = 3.0332e+53 * 1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*kpc^(-3)*keV^(-1)
  // Absorb factors of 1e-8 from n_e and n_i directly into the prefactor:
  // => Brems_per_energy = 3.0332e+37 * 1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*kpc^(-3)*keV^(-1)
  // Convert to flux (per m^2) by assuming a distance r in kpc:
  // => Brems_flux = 1/(r [kpc])^2 * 1/(4*pi*(1 kpc [m])^2)*3.0332e+37*1/sqrt(T)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*kpc^(-3)*keV^(-1)
  //               = 0.002535f * 1/sqrt(T)*(r [kpc])^(-2)*g_ff*n_e*n_i*Z^2*exp(-E/(kT)) keV*s^(-1)*kpc^(-3)*keV^(-1)*m^(-2)
  scalar r2 = r*r;
  scalar emission = 0.002535f * dE * Z*Z * ion_density*ion_density
      * g_ff(E,T,Z,gaunt_table) * rsqrt(T) * exp(-E * T_keV_inv) / r2; // rsqrt computes inverse square root

  return emission;
}

/// \return The thermal Bremsstrahlung X-Ray count rate within an
/// energy bin in units of 1/s/kpc^3/m^2
///  for a hydrogen plasma
/// \param E the photon energy in keV
/// \param dE energy bin width in keV
/// \param T the plasma temperature in K
/// \param electron_abundance the electron abundance (dimensionless)
/// \param density the density in units of M_sun/kpc^3
/// \param r the distance to the observer in kpc (luminosity distance)
/// \param gaunt_table tabulated gaunt factors
scalar get_spectral_brems_count_rate(scalar E,
                                     scalar dE,
                                     scalar T,
                                     scalar electron_abundance,
                                     scalar density,
                                     scalar r,
                                     __read_only image2d_t gaunt_table)
{
  return get_spectral_brems_emission(E,dE,T,electron_abundance,density,r,gaunt_table) / E;
}

/// Calculates the temperature for each data element
/// \param out output array for temperatures in K
/// \param num_elements number of data elements in input (and output) arrays
/// \param densities density in M_sun/kpc^3
/// \param internal_energies specific internal energy in km^2/s^2
/// \param electron_abundance The electron abundance (dimensionless)
__kernel void mean_temperature(
                            __global scalar* out,
                            unsigned num_elements,
                            __global scalar* densities,
                            __global scalar* internal_energies,
                            __global scalar* electron_abundances)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];

    out[tid] = get_temperature(internal_energy, electron_abundance);
  }
}

/* ToDo: Fix this
__kernel void luminosity_weighted_temperature(
                            __global scalar* out,
                            unsigned num_elements,
                            __global scalar* densities,
                            __global scalar* internal_energies,
                            __global scalar* electron_abundances)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];
    scalar T = get_temperature(internal_energy, electron_abundance);

    out[tid] = density * density * sqrt(T) * T;
  }
}
*/


/// Calculates the X-Ray flux within a given energy range by
/// integrating over a number of energy bins. The result will be
/// in units of keV/s/kpc^3/m^2
/// \param out Output array, must be at least of size \c num_elements
/// \param num_elements number of data elements to process
/// \param densities The densities in M_sun/kpc^3
/// \param internal_energies The specific internal energies in (km/s)^2
/// \param electron_abundances The electron abundances (dimensionless)
/// \param z The redshift of the observed object
/// \param luminosity_distance The luminosity distance of the observed
/// object in kpc
/// \param gaunt_table The tabulated thermally averaged gaunt factors
/// \param E_min minimum integration energy in keV
/// \param E_max maximum integration energy in keV
/// \param num_samples The number of samples for the integration
__kernel void xray_flux(__global scalar* out,
                            unsigned num_elements,
                            __global scalar* densities,
                            __global scalar* internal_energies,
                            __global scalar* electron_abundances,
                            scalar z,
                            scalar luminosity_distance,
                            __read_only image2d_t gaunt_table,
                            scalar E_min,
                            scalar E_max,
                            int num_samples)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];
    scalar T = get_temperature(internal_energy, electron_abundance);

    scalar dE = (E_max - E_min) / num_samples;

    scalar emission = 0.0f;
    for(int i = 0; i < num_samples; ++i)
    {
      scalar current_photon_energy = E_min + (i + 0.5f) * dE;

      emission += get_spectral_brems_emission((1.f + z) * current_photon_energy,
                                              (1.f + z) * dE,
                                              T,
                                              electron_abundance,
                                              density,
                                              luminosity_distance,
                                              gaunt_table);

    }

    out[tid] = emission;
  }
}

/// Calculates the X-Ray flux within a given energy bin. The result will be
/// in units of keV/s/kpc^3/m^2
/// \param out Output array, must be at least of size \c num_elements
/// \param num_elements number of data elements to process
/// \param densities The densities in M_sun/kpc^3
/// \param internal_energies The specific internal energies in (km/s)^2
/// \param electron_abundances The electron abundances (dimensionless)
/// \param z The redshift of the observed object
/// \param luminosity_distance The luminosity distance of the observed
/// object in kpc
/// \param gaunt_table The tabulated thermally averaged gaunt factors
/// \param E the photon energy
/// \param dE the photon energy bin width
__kernel void xray_spectral_flux(__global scalar* out,
                            unsigned num_elements,
                            __global scalar* densities,
                            __global scalar* internal_energies,
                            __global scalar* electron_abundances,
                            scalar z,
                            scalar luminosity_distance,
                            __read_only image2d_t gaunt_table,
                            scalar E,
                            scalar dE)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];
    scalar T = get_temperature(internal_energy, electron_abundance);


    out[tid] = get_spectral_brems_emission((1.f + z) * E,
                                           (1.f + z) * dE,
                                           T,
                                           electron_abundance,
                                           density,
                                           luminosity_distance,
                                           gaunt_table);

  }
}


/// Calculates the total count rate, as observed by the chandra
/// observatory in units of 1/s/kpc^3 (the kpc^3 must be integrated
/// away by a 3D integration over the observed object)
/// \param out Output array, must be at least of size \c num_elements
/// \param num_elements number of data elements to process
/// \param densities The densities in M_sun/kpc^3
/// \param internal_energies The specific internal energies in (km/s)^2
/// \param electron_abundances The electron abundances (dimensionless)
/// \param z The redshift of the observed object
/// \param luminosity_distance The luminosity distance of the observed
/// object in kpc
/// \param gaunt_table The tabulated thermally averaged gaunt factors
/// \param arf_table The tabulated chandra ARF
/// \param arf_min_energy The energy of the first value of the ARF table in keV
/// \param arf_num_energy_bins The number of of entries in the ARF table,
/// i.e. the number of energy channels
/// \param arf_energy_bin_width The energy spacing between each entry
/// in the ARF table (in keV)
__kernel void chandra_xray_total_count_rate(__global scalar* out,
                                            unsigned num_elements,
                                            __global scalar* densities,
                                            __global scalar* internal_energies,
                                            __global scalar* electron_abundances,
                                            scalar z,
                                            scalar luminosity_distance,
                                            __read_only image2d_t gaunt_table,
                                            __read_only image1d_t arf_table,
                                            scalar arf_min_energy,
                                            int    arf_num_energy_bins,
                                            scalar arf_energy_bin_width)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];
    scalar T = get_temperature(internal_energy, electron_abundance);

    scalar emission = 0.0f;

    // Integrate over the arf
    for(int i = 0; i < arf_num_energy_bins; ++i)
    {
      scalar current_photon_energy = arf_min_energy + i * arf_energy_bin_width;

      scalar arf_value = evaluate_tabulated_function(arf_table,
                                                     arf_min_energy,
                                                     arf_energy_bin_width,
                                                     current_photon_energy);

      emission += get_spectral_brems_count_rate((1.f + z)*current_photon_energy,
                                                (1.f + z)*arf_energy_bin_width,
                                                T,
                                                electron_abundance,
                                                density,
                                                luminosity_distance,
                                                gaunt_table) * arf_value;

    }

    out[tid] = emission;
  }
}

/// Calculates the total count rate, as observed by the chandra
/// observatory, in units of 1/s/kpc^3 (the kpc^3 must be integrated
/// away by a 3D integration over the observed object), within an energy
/// bin.
/// \param out Output array, must be at least of size \c num_elements
/// \param num_elements number of data elements to process
/// \param densities The densities in M_sun/kpc^3
/// \param internal_energies The specific internal energies in (km/s)^2
/// \param electron_abundances The electron abundances (dimensionless)
/// \param z The redshift of the observed object
/// \param luminosity_distance The luminosity distance of the observed
/// object in kpc
/// \param photon_energy The energy in keV at which the count rate shall
/// be determined
/// \param photon_energy_bin_width The energy range around \c photon_energy
/// in keV that shall be included (Remember that absolute count rates can
/// only be determined for a fixed energy range - otherwise only a differential
/// quantity can be obtained)
/// \param gaunt_table The tabulated thermally averaged gaunt factors
/// \param arf_table The tabulated chandra ARF
/// \param arf_min_energy The energy of the first value of the ARF table in keV
/// \param arf_num_energy_bins The number of of entries in the ARF table,
/// i.e. the number of energy channels
/// \param arf_energy_bin_width The energy spacing between each entry
/// in the ARF table (in keV)
__kernel void chandra_xray_spectral_count_rate(__global scalar* out,
                                            unsigned num_elements,
                                            __global scalar* densities,
                                            __global scalar* internal_energies,
                                            __global scalar* electron_abundances,
                                            scalar z,
                                            scalar luminosity_distance,
                                            __read_only image2d_t gaunt_table,
                                            __read_only image1d_t arf_table,
                                            scalar arf_min_energy,
                                            int    arf_num_energy_bins,
                                            scalar arf_energy_bin_width,
                                            scalar photon_energy,
                                            scalar photon_energy_bin_width)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];
    scalar T = get_temperature(internal_energy, electron_abundance);

    scalar arf_value = evaluate_tabulated_function(arf_table,
                                                   arf_min_energy,
                                                   arf_energy_bin_width,
                                                   photon_energy);

    out[tid] = get_spectral_brems_count_rate( (1.f + z)*photon_energy,
                                              (1.f + z)*photon_energy_bin_width,
                                              T,
                                              electron_abundance,
                                              density,
                                              luminosity_distance,
                                              gaunt_table) * arf_value;


  }
}

/// Sets the output to 1.f for all elements. Only usefule for
/// testing the interpolation quality.
/// \param out Output array, with at least \c num_elements elements
/// \param num_elements The number of elements
/// \param densities The densities - will not be used
/// \param internal_energies The internal energies - unused
__kernel void identity(__global scalar* out,
                       unsigned num_elements,
                       __global scalar* densities,
                       __global scalar* internal_energies)
{
  int tid = get_global_id(0);

  if(tid < num_elements)
  {
    out[tid] = 1.f;
  }
}



#endif
