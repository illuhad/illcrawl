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

#ifndef TILES_CL
#define TILES_CL

#include "types.cl"

int3 get_tile_around_pos3(vector3 coordinate,
                      vector3 tiles_min_corner,
                      vector3 tile_sizes)
{
  vector3 float_result = (coordinate - tiles_min_corner) / tile_sizes;

  return (int3)((int)floor(float_result.x),
                (int)floor(float_result.y),
                (int)floor(float_result.z));
}

ulong get_tile_index(vector3 coordinate,
                     vector3 tiles_min_corner,
                     vector3 tile_sizes,
                     int3 num_tiles)
{
  int3 grid_cell = get_tile_around_pos3(coordinate,
                                        tiles_min_corner,
                                        tile_sizes);

  ulong index = (ulong)grid_cell.x
              + (ulong)grid_cell.y * (ulong)num_tiles.x
              + (ulong)grid_cell.z * (ulong)num_tiles.x * (ulong)num_tiles.y;

  return index;
}

__kernel void generate_sort_keys(__global vector4* particles,
                                 __global ulong* keys_out,
                                 int3 num_tiles,
                                 vector3 tiles_min_corner,
                                 vector3 tile_sizes,
                                 int num_particles)
{
  int tid = get_global_id(0);

  if(tid < num_particles)
  {
    keys_out[tid] = get_tile_index(particles[tid].xyz,
                               tiles_min_corner,
                               tile_sizes,
                               num_tiles);
  }
}

__kernel void generate_tiles(__global vector4* sorted_particles,
                             __global int2* tiles_out,
                             vector3 tiles_min_corner,
                             vector3 tile_sizes,
                             int3 num_tiles)
{
  int tid = get_global_id(0);

  if(tid < num_particles)
  {
    vector4 current_particle = sorted_particles[tid];

    ulong tile_index = get_tile_index(current_particle.xyz,
                                      tiles_min_corner,
                                      tile_sizes,
                                      num_tiles);

    int is_first_particle_of_tile = false;
    int is_last_particle_of_tile = false;

    if(tid > 0)
    {
      vector4 prev_particle = sorted_particles[tid - 1];
      ulong prev_tile_index = get_tile_index(prev_particle.xyz,
                                             tiles_min_corner,
                                             tile_sizes,
                                             num_tiles);
      if(prev_tile_index != tile_index)
        is_first_particle_of_tile = true;
    }

    if(tid < num_paricles - 1)
    {
      vector4 next_particle = sorted_particles[tid + 1];
      ulong next_tile_index = get_tile_index(next_particle.xyz,
                                             tiles_min_corner,
                                             tile_sizes,
                                             num_tiles);
      if(next_tile_index != tile_index)
        is_last_particle_of_tile = true;
    }

    if(is_first_particle_of_tile)
      tiles[tile_index].x = tid;

    if(is_last_particle_of_tile)
      tiles[tile_index].y = tid + 1;
  }
}

#endif
