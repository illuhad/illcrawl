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


#ifndef RECONSTRUCTION_CL
#define RECONSTRUCTION_CL

#include "interpolation.cl"



int2 get_tile_around_pos(vector2 coordinate,
                      vector2 tiles_min_corner,
                      vector2 tile_size)
{
  vector2 float_result = (coordinate - tiles_min_corner) / tile_size;

  return (int2)((int)float_result.x, (int)float_result.y);
}


// CLK_ADDRESS_CLAMP will return (vector4)(0.0f) outside the bounds of
// the image for CL_RGBA images.
//const sampler_t tile_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;

__kernel void image_tile_based_reconstruction2D(__global vector4* tiles,
                                                __global vector4* particles,
                                                scalar maximum_smoothing_length,
                                                __global scalar* pixels_out,
                                                vector2 pixels_min_corner,
                                                vector2 pixels_max_corner,
                                                unsigned num_px_x,
                                                unsigned num_px_y,
                                                unsigned num_tiles_x,
                                                unsigned num_tiles_y,
                                                //__local vector4* local_mem,
                                                //__local scalar* smoothing_length_cache,
                                                __global scalar* quantity)
{
  int tid_x = get_global_id(0);
  int tid_y = get_global_id(1);

  //int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
  //int total_local_size = get_local_size(0)*get_local_size(1);
  //int depth = get_image_depth(tiles);

  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);

  scalar dx = (pixels_max_corner.x - pixels_min_corner.x) / (float)num_px_x;
  scalar dy = (pixels_max_corner.y - pixels_min_corner.y) / (float)num_px_y;

  vector2 tile_min = pixels_min_corner;
  tile_min.x += get_group_id(0) * get_local_size(0) * dx;
  tile_min.y += get_group_id(1) * get_local_size(1) * dy;

  vector2 tile_size;
  tile_size.x = get_local_size(0) * dx;
  tile_size.y = get_local_size(1) * dy;

  vector2 tile_max = tile_min;
  tile_max.x += tile_size.x * dx;
  tile_max.y += tile_size.y * dy;

  vector2 tile_center = 0.5f * (tile_min + tile_max);
  scalar tile_radius = 0.5f * sqrt(tile_size.x*tile_size.x + tile_size.y*tile_size.y);

  vector2 pixel_min = tile_min;
  pixel_min.x += get_local_id(0) * dx;
  pixel_min.y += get_local_id(1) * dy;

  vector2 pixel_max = pixel_min;
  pixel_max.x += dx;
  pixel_max.y += dy;

  vector2 px_center = 0.5f * (pixel_min + pixel_max);

  int2 min_tile = get_tile_around_pos(tile_center - (maximum_smoothing_length + tile_radius),
                                      pixels_min_corner,
                                      tile_size);
  min_tile.x = max(0, min_tile.x);
  min_tile.y = max(0, min_tile.y);

  int2 max_tile = get_tile_around_pos(tile_center + (maximum_smoothing_length + tile_radius),
                                      pixels_min_corner,
                                      tile_size);

  max_tile.x = min((int)(num_tiles_x-1), max_tile.x);
  max_tile.y = min((int)(num_tiles_y-1), max_tile.y);


  //if(get_global_id(0)==512 && get_global_id(1)==512)
  //  printf("center %f %f tile_min %f %f min_tile %d %d max_tile %d %d\n",tile_center.x,tile_center.y,tile_min.x,tile_min.y,min_tile.x,min_tile.y,max_tile.x,max_tile.y);
  scalar result = 0.0f;
  for(int current_tile_x = min_tile.x; current_tile_x <= max_tile.x; ++current_tile_x)
  {
    for(int current_tile_y = min_tile.y; current_tile_y <= max_tile.y; ++current_tile_y)
    {
      // Retrieve tile header
      int2 tile_coord = (int2)(current_tile_x, current_tile_y);
      vector4 header = tiles[tile_coord.y * num_tiles_x + tile_coord.x];

      int num_particles_in_tile = (int)header.x;
      scalar maximum_smoothing_distance_of_tile = header.y;
      int particle_data_offset = (int)header.z;

      scalar r = maximum_smoothing_distance_of_tile + tile_radius;
      vector2 current_tile_center = pixels_min_corner;
      current_tile_center.x += (current_tile_x + 0.5f) * tile_size.x;
      current_tile_center.y += (current_tile_y + 0.5f) * tile_size.y;

      if (distance2(current_tile_center, px_center) < r * r)
      {

        //for (int i = 0; i < num_particles_in_tile; i += total_local_size)
        for (int i = 0; i < num_particles_in_tile; ++i)
        {
          //if(i+local_id < num_particles_in_tile)
          //  local_mem[local_id] = particles[particle_data_offset + i + local_id];
          //barrier(CLK_LOCAL_MEM_FENCE);

          //for(int j = 0; i+j < num_particles_in_tile; ++j)
          //{
          //  vector4 particle = local_mem[j];
          vector4 particle = particles[particle_data_offset + i];
          scalar smoothing_length = particle.z;
          int particle_id = (int)particle.w;

          scalar weight = get_weight(particle.xy, smoothing_length, px_center,
                                     dx, dy, pixel_min, pixel_max);

          // if(get_global_id(0)==512 && get_global_id(1)==512)
          //  printf("%f %f %f %d %d\n", weight, particle.x, particle.y,
          //  current_tile_x, current_tile_y);
          if (weight != 0.0f)
            result += weight * quantity[particle_id];

        }
      }
    }
  }

  pixels_out[tid_y * num_px_x + tid_x] += result;
}



#endif
