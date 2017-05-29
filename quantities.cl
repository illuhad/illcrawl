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

// Boltzmann constant in eV/K
#define k_B 8.61733e-5f
// adiabatic index
#define gamma (5.f/3.f)

// Returns the mean molecular weight in units of proton masses
scalar mean_molecular_weight(scalar electron_abundance)
{
  return 4.f/(1.f + 3*X_H + 4*X_H*electron_abundance) /* * m_p*/;
}

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
    scalar density = densities[tid];
    scalar internal_energy = internal_energies[tid];
    scalar electron_abundance = electron_abundances[tid];

    out[tid] = get_temperature(internal_energy, electron_abundance);
  }
}

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

__kernel void xray_emission(__global scalar* out,
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

    scalar electron_density = density * electron_abundance;

    out[tid] = electron_density * electron_density * sqrt(T);
  }
}

__kernel void chandra_xray_emission(__global scalar* out,
                                    unsigned num_elements,
                                    __global scalar* densities,
                                    __global scalar* internal_energies,
                                    __global scalar* electron_abundances,
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
    scalar sqrt_T_inv = 1.0f / sqrt(T);

    scalar electron_density = density * electron_abundance;

    scalar emission = 0.0f;

    // Integrate over the arf
    for(int i = 0; i < arf_num_energy_bins; ++i)
    {
      // The arf min energy and the arf bin width is assumed to be in units
      // of keV.
      scalar current_photon_energy = arf_min_energy + i * arf_energy_bin_width;
      scalar arf_value = evaluate_tabulated_function(arf_table,
                                                     arf_min_energy,
                                                     arf_energy_bin_width,
                                                     current_photon_energy);

      // Calculate emission according to
      // L ~ n_i*n_e/sqrt(T)*exp(-h*nu/(k_B*T))
      // The factor of 1000 converts the current photon energy from keV to eV.
      emission += electron_density * electron_density
                * sqrt_T_inv * exp(-current_photon_energy * 1000 / (k_B * T)) * arf_value;

    }

    out[tid] = emission;
  }
}

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
