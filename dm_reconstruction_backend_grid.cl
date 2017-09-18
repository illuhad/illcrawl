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

#ifndef DM_RECONSTRUCTION_BACKEND_GRID_CL
#define DM_RECONSTRUCTION_BACKEND_GRID_CL

#include "smoothing_kernel.cl"
#include "particle_grid.cl"
#include "types.cl"

#ifdef USE_CUBIC_SPLINE
 #define CUTOFF_RADIUS(r) cubic_spline3d_cutoff_radius(r)
 #define SMOOTHING_KERNEL(r,h) cubic_spline3d(r,h)
#else
 #define CUTOFF_RADIUS(r) quartic_polynomial3d_cutoff_radius(r)
 #define SMOOTHING_KERNEL(r,h) quartic_polynomial3d(r,h)
#endif

__kernel void dm_reconstruction_grid_smoothing(
    __global int2* grid_cells,
    int3 num_grid_cells,
    vector3 grid_min_corner,
    vector3 grid_cell_sizes,

    __global vector4* particles,
    __global scalar* smoothing_lengths,
    scalar overall_max_smoothing_length,
    __global scalar* max_smoothing_lengths,

    int num_evaluation_points,
    __global vector4* evaluation_points_coordinates,
    __global scalar* results)
{
  size_t tid = get_global_id(0);

  scalar cutoff_radius = CUTOFF_RADIUS(overall_max_smoothing_length);

  if(tid < num_evaluation_points)
  {
    scalar result = results[tid];

    vector4 evaluation_point = evaluation_points_coordinates[tid];

    scalar cell_radius = 0.5f * sqrt(dot(grid_cell_sizes, grid_cell_sizes));

    grid3d_ctx grid;
    grid3d_init(&grid, grid_min_corner, grid_cell_sizes, num_grid_cells);


    int3 min_grid_cell = grid3d_get_cell_indices(&grid,
                                                 evaluation_point.xyz - (vector3)cutoff_radius);

    int3 max_grid_cell = grid3d_get_cell_indices(&grid,
                                                 evaluation_point.xyz + (vector3)cutoff_radius);


    int3 current_cell;
    for(current_cell.z =  min_grid_cell.z;
        current_cell.z <= max_grid_cell.z;
        ++current_cell.z)
    {
      for(current_cell.y =  min_grid_cell.y;
          current_cell.y <= max_grid_cell.y;
          ++current_cell.y)
      {
        for(current_cell.x =  min_grid_cell.x;
            current_cell.x <= max_grid_cell.x;
            ++current_cell.x)
        {
          if(grid3d_is_cell_in_grid(&grid, current_cell))
          {
            ulong cell_key = grid3d_get_cell_key_from_indices(&grid, current_cell);
            int2 cell_entry = grid_cells[cell_key];
            scalar cell_cutoff =
                 CUTOFF_RADIUS(max_smoothing_lengths[cell_key]);

            vector3 cell_center = grid3d_get_cell_center(&grid, current_cell);
            if(distance(evaluation_point.xyz, cell_center)
               < (cell_cutoff + cell_radius))
            {

              for(int current_particle_id = cell_entry.x;
                  current_particle_id < cell_entry.y;
                  ++current_particle_id)
              {
                scalar smoothing_length = smoothing_lengths[current_particle_id];
                vector4 particle = particles[current_particle_id];

                scalar r = distance(particle.xyz, evaluation_point.xyz);

                result += particle.w * SMOOTHING_KERNEL(r, smoothing_length);
              }
            }
          }
        }
      }
    }

    results[tid] = result;
  }

}

#endif

