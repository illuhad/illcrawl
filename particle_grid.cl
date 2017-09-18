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

#ifndef PARTICLE_GRID_CL
#define PARTICLE_GRID_CL

#include "types.cl"
#include "util.cl"

typedef struct
{
  vector3 min_corner;
  vector3 cell_sizes;
  int3 num_cells;
} grid3d_ctx;

void grid3d_init(grid3d_ctx* ctx,
                 vector3 min_corner,
                 vector3 cell_sizes,
                 int3 num_cells)
{
  ctx->min_corner = min_corner;
  ctx->cell_sizes = cell_sizes;
  ctx->num_cells = num_cells;
}


int3 grid3d_get_cell_indices(grid3d_ctx* ctx,
                             vector3 point)
{
  vector3 float_result = (point - ctx->min_corner) / ctx->cell_sizes;

  return (int3)((int)floor(float_result.x),
                (int)floor(float_result.y),
                (int)floor(float_result.z));
}

vector3 grid3d_get_cell_min_corner(grid3d_ctx* ctx,
                                   int3 cell_index)
{
  return ctx->min_corner + (vector3)((float)cell_index.x,
                                     (float)cell_index.y,
                                     (float)cell_index.z) * ctx->cell_sizes;
}

vector3 grid3d_get_cell_center(grid3d_ctx* ctx,
                               int3 cell_index)
{
  vector3 min_corner = grid3d_get_cell_min_corner(ctx, cell_index);
  return min_corner + 0.5f * ctx->cell_sizes;
}

int grid3d_is_cell_in_grid(grid3d_ctx* ctx,
                           int3 cell_index)
{
  int3 result = (0 <= cell_index && cell_index < ctx->num_cells);
  return result.x && result.y && result.z;
}

ulong grid3d_get_cell_key_from_indices(grid3d_ctx* ctx,
                                       int3 grid_cell)
{
  ulong key = (ulong)grid_cell.x
            + (ulong)grid_cell.y * (ulong)(ctx->num_cells.x)
            + (ulong)grid_cell.z * (ulong)(ctx->num_cells.x) * (ulong)(ctx->num_cells.y);

  return key;
}

ulong grid3d_get_cell_key(grid3d_ctx* ctx,
                         vector3 point)
{
  int3 grid_cell = grid3d_get_cell_indices(ctx, point);

  return grid3d_get_cell_key_from_indices(ctx, grid_cell);
}


__kernel void grid3d_generate_sort_keys(__global vector4* particles,
                                 __global ulong* keys_out,
                                 int3 num_cells,
                                 vector3 grid_min_corner,
                                 vector3 grid_cell_sizes,
                                 int num_particles)
{
  int tid = get_global_id(0);

  if(tid < num_particles)
  {
    grid3d_ctx grid;
    grid3d_init(&grid, grid_min_corner, grid_cell_sizes, num_cells);

    keys_out[tid] = grid3d_get_cell_key(&grid, particles[tid].xyz);
  }
}

__kernel void grid3d_determine_cells_begin(__global vector4* sorted_particles,
                                           int num_particles,
                                           __global int2* cells_out,
                                           vector3 grid_min_corner,
                                           vector3 grid_cell_sizes,
                                           int3 num_cells)
{
  int tid = get_global_id(0);

  if(tid < num_particles)
  {
    grid3d_ctx grid;
    grid3d_init(&grid, grid_min_corner, grid_cell_sizes, num_cells);

    vector4 current_particle = sorted_particles[tid];
    ulong cell_key = grid3d_get_cell_key(&grid, current_particle.xyz);

    int is_first_particle_of_cell = false;

    if(tid > 0)
    {
      vector4 prev_particle = sorted_particles[tid - 1];
      ulong prev_cell_key = grid3d_get_cell_key(&grid, prev_particle.xyz);

      if(prev_cell_key != cell_key)
        is_first_particle_of_cell = true;
    }
    else
    {
      // The first particle of all is always the first particle of a cell
      is_first_particle_of_cell = true;
    }


    if(is_first_particle_of_cell)
      cells_out[cell_key].x = tid;

  }
}

__kernel void grid3d_determine_cells_end(__global vector4* sorted_particles,
                                         int num_particles,
                                         __global int2* cells_out,
                                         vector3 grid_min_corner,
                                         vector3 grid_cell_sizes,
                                         int3 num_cells)
{
  int tid = get_global_id(0);

  if(tid < num_particles)
  {
    grid3d_ctx grid;
    grid3d_init(&grid, grid_min_corner, grid_cell_sizes, num_cells);

    vector4 current_particle = sorted_particles[tid];
    ulong cell_key = grid3d_get_cell_key(&grid, current_particle.xyz);

    int is_last_particle_of_cell = false;

    if(tid < num_particles - 1)
    {
      vector4 next_particle = sorted_particles[tid + 1];
      ulong next_cell_key = grid3d_get_cell_key(&grid, next_particle.xyz);

      if(next_cell_key != cell_key)
        is_last_particle_of_cell = true;
    }
    else
    {
      // The last particle of all is always the last particle of a cell
      is_last_particle_of_cell = true;
    }

    if(is_last_particle_of_cell)
      cells_out[cell_key].y = tid + 1;
  }
}

DEFINE_APPLY_PERMUTATION_KERNEL(grid3d_sort_particles_into_cells, vector4);
DEFINE_APPLY_PERMUTATION_KERNEL(grid3d_sort_scalars_into_cells, scalar);

__kernel void grid3d_determine_max_per_cell(__global int2* cells,
                                            __global scalar* data,
                                            __global scalar* out,
                                            unsigned long num_cells)
{
  size_t tid = get_global_id(0);

  if(tid < num_cells)
  {
    int2 cell = cells[tid];
    scalar current_max = 0.0f;

    for(int i = cell.x; i < cell.y; ++i)
    {
      scalar current_value = data[i];
      if(current_value > current_max)
        current_max = current_value;
    }

    out[tid] = current_max;
  }
}



#endif
