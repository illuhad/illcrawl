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

#define X_H 0.76f

scalar mean_molecular_weight(scalar electron_abundance)
{
  return 4.f/(1.f + 3*X_H + 4*X_H*electron_abundance) /* * m_p*/;
}

scalar get_temperature(scalar internal_energy, scalar electron_abundance)
{
  return mean_molecular_weight(electron_abundance) * internal_energy;
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
