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

#ifndef UTIL_CL
#define UTIL_CL

__kernel void util_create_sequence(__global unsigned long* output_buffer,
                                   unsigned long begin,
                                   unsigned long num_elements)
{
  size_t tid = get_global_id(0);

  if(tid < num_elements)
  {
    output_buffer[tid] = begin + tid;
  }
}

#define DEFINE_APPLY_PERMUTATION_KERNEL(kernel_name, T)        \
__kernel void kernel_name(__global T* input_buffer,            \
                          __global T* output_buffer,           \
                          unsigned long num_elements,          \
                          __global unsigned long* permutation) \
{                                                              \
  size_t tid = get_global_id(0);                               \
  if(tid < num_elements)                                       \
  {                                                            \
    output_buffer[tid] = input_buffer[permutation[tid]];       \
  }                                                            \
}


#endif
