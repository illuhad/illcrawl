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

#ifndef MATH_CL
#define MATH_CL

#include "types.cl"

typedef struct
{
  vector3 row0;
  vector3 row1;
  vector3 row2;
} matrix3x3_t;

#define M00(m_ptr) ((m_ptr)->row0.x)
#define M10(m_ptr) ((m_ptr)->row1.x)
#define M20(m_ptr) ((m_ptr)->row2.x)

#define M01(m_ptr) ((m_ptr)->row0.y)
#define M11(m_ptr) ((m_ptr)->row1.y)
#define M21(m_ptr) ((m_ptr)->row2.y)

#define M02(m_ptr) ((m_ptr)->row0.z)
#define M12(m_ptr) ((m_ptr)->row1.z)
#define M22(m_ptr) ((m_ptr)->row2.z)

vector3 matrix3x3_vector_mult(matrix3x3_t* m,
                              vector3 v)
{
  vector3 result;
  result.x = dot(m->row0, v);
  result.y = dot(m->row1, v);
  result.z = dot(m->row2, v);
  return result;
}

scalar matrix3x3_det(matrix3x3_t* m)
{
  scalar result = 0.0f;

  result += M00(m) * M11(m) * M22(m);
  result += M01(m) * M12(m) * M20(m);
  result += M02(m) * M10(m) * M21(m);

  result -= M20(m) * M11(m) * M02(m);
  result -= M21(m) * M12(m) * M00(m);
  result -= M22(m) * M10(m) * M01(m);

  return result;
}

void matrix3x3_invert(matrix3x3_t* m)
{
  scalar inv_detA = 1.f / matrix3x3_det(m);

  scalar a = M00(m);
  scalar b = M01(m);
  scalar c = M02(m);
  scalar d = M10(m);
  scalar e = M11(m);
  scalar f = M12(m);
  scalar g = M20(m);
  scalar h = M21(m);
  scalar i = M22(m);

  m->row0 = (vector3)(e*i - f*h, c*h - b*i, b*f - c*e);
  m->row1 = (vector3)(f*g - d*i, a*i - c*g, c*d - a*f);
  m->row2 = (vector3)(d*h - e*g, b*g - a*h, a*e - b*d);

  m->row0 *= inv_detA;
  m->row1 *= inv_detA;
  m->row2 *= inv_detA;
}

/// x <- alpha*x
__kernel void vector_scale(__global scalar* x,
                           scalar alpha,
                           unsigned long num_elements)
{
  size_t tid = get_global_id(0);

  if(tid < num_elements)
    x[tid] *= alpha;
}

/// x <- x + alpha*y
__kernel void vector_scale_add(__global scalar* x,
                               scalar alpha,
                               __global scalar* y,
                               unsigned long num_elements)
{
  size_t tid = get_global_id(0);

  if(tid < num_elements)
    x[tid] += alpha * y[tid];
}

/// x <- alpha*x + y
__kernel void vector_axpy(scalar alpha,
                          __global scalar* x,
                          __global scalar* y,
                          unsigned long num_elements)
{
  size_t tid = get_global_id(0);

  if(tid < num_elements)
    x[tid] = alpha*x[tid] + y[tid];
}

#endif


