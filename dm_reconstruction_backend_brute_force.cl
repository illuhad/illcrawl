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

#ifndef DM_RECONSTRUCTION_BRUTE_FORCE
#define DM_RECONSTRUCTION_BRUTE_FORCE

#include "smoothing_kernel.cl"
#include "types.cl"

__kernel void dm_reconstruction_brute_force_smoothing(__global vector4* particles,
                                                      __global scalar* smoothing_lengths,
                                                      unsigned num_particles,
                                                      __local vector4* particle_cache,
                                                      __local scalar* smoothing_length_cache,
                                                      __global vector4* evaluation_points,
                                                      unsigned num_evaluation_points,
                                                      __global scalar* result_buffer)
{
  size_t tid = get_global_id(0);
  size_t lid = get_local_id(0);
  size_t global_size = get_global_size(0);
  size_t group_size = get_local_size(0);


  vector4 evaluation_point = (vector4)(0);
  scalar result = 0.0f;

  if(tid < num_evaluation_points)
  {
    evaluation_point = evaluation_points[tid];
    result = result_buffer[tid];
  }

  // collectively load particle chunk into local memory
  for(size_t particle_offset = 0;
      particle_offset < num_particles;
      particle_offset += group_size)
  {
    size_t particle = lid + particle_offset;

    if(particle < num_particles)
    {
      particle_cache[lid]         = particles[particle];
      smoothing_length_cache[lid] = smoothing_lengths[particle];
    }
    else
    {
      particle_cache[lid] = (vector4)(0.f);
      smoothing_length_cache[lid] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < group_size; ++i)
    {
      vector4 current_particle = particle_cache[i];
      scalar  current_smoothing_length = smoothing_length_cache[i];

      scalar r = distance(current_particle.xyz, evaluation_point.xyz);
      // Smoothing kernel weight
      scalar W = cubic_spline3d(r, current_smoothing_length);
      // Particle "mass" is stored in current_particle.w
      result += current_particle.w * W;
    }
  }

  if(tid < num_evaluation_points)
  {
    result_buffer[tid] = result;
  }

}

#endif
