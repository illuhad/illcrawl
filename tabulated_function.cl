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

#ifndef TABULATED_FUNCTION_CL
#define TABULATED_FUNCTION_CL

#include "types.cl"

__constant sampler_t linear_sampler =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_LINEAR;

scalar evaluate_tabulated_function(__read_only image1d_t evaluation_table,
                                   scalar x_min,
                                   scalar dx,
                                   scalar position)
{
  float rel_pos = (position - x_min) / dx;

  float4 function_value = read_imagef(evaluation_table,
                                      linear_sampler,
                                      rel_pos);
  // Image format is CL_R, therefore the value will be stored
  // in the first component
  return function_value.s0;
}

scalar evaluate_tabulated_function2d(__read_only image2d_t evaluation_table,
                                     vector2 xy_min,
                                     vector2 delta,
                                     vector2 position)
{
  vector2 rel_pos = (position - xy_min) / delta;

  float4 function_value = read_imagef(evaluation_table,
                                      linear_sampler,
                                      rel_pos);
  // Image format is CL_R, therefore the value will be stored
  // in the first component
  return function_value.s0;
}

#endif
