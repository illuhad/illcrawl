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

#include "types.cl"

__kernel void camera_generate_pixel_coordinates(
                                       __global vector4* out,
                                       unsigned long num_pix_x,
                                       unsigned long num_pix_y,
                                       scalar pixel_size,
                                       vector4 camera_min_corner,
                                       vector4 camera_basis0,
                                       vector4 camera_basis1)
{
  size_t tid_x = get_global_id(0);
  size_t tid_y = get_global_id(1);

  if(tid_x < num_pix_x && tid_y < num_pix_y)
  {
    out[tid_y * num_pix_x + tid_x] = camera_min_corner
                                   + tid_x * pixel_size * camera_basis0
                                   + tid_y * pixel_size * camera_basis1;
  }
}
