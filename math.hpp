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


#ifndef MATH_HPP
#define MATH_HPP

#include <array>
#include <cmath>

#include "types.hpp"

#ifndef __OPENCL_VERSION__
#define VECTOR3_X(vector) (vector[0])
#define VECTOR3_Y(vector) (vector[1])
#define VECTOR3_Z(vector) (vector[2])
#define VECTOR3(x, y, z) {{x, y, z}}
#else
#define VECTOR3_X(vector) (vector.s0)
#define VECTOR3_Y(vector) (vector.s1)
#define VECTOR3_Z(vector) (vector.s2)
#define VECTOR3(x,y,z) (vector3)(x, y, z)
#endif

namespace illcrawl {
namespace math {

std::size_t make_multiple_of(std::size_t n, std::size_t value)
{
  std::size_t result = (value / n) * n;
  if(result != value)
    result += n;

  return result;
}


using scalar = double;

template<std::size_t Dim>
using vector_n = std::array<scalar, Dim>;

using vector3 = vector_n<3>;
using vector2 = vector_n<2>;

inline
device_vector3 to_device_vector3(const math::vector3& v)
{
  return device_vector3{{static_cast<device_scalar>(v[0]),
                         static_cast<device_scalar>(v[1]),
                         static_cast<device_scalar>(v[2])}};
}

inline
device_vector4 to_device_vector4(const math::vector3& v)
{
  return device_vector4{{static_cast<device_scalar>(v[0]),
                         static_cast<device_scalar>(v[1]),
                         static_cast<device_scalar>(v[2]),
                         device_scalar{}}};
}

struct matrix3x3
{
  vector3 row0;
  vector3 row1;
  vector3 row2;
};

template<std::size_t Dim>
inline
scalar dot(const vector_n<Dim>& a, const vector_n<Dim>& b)
{
  scalar result = 0.0;
  for(std::size_t i = 0; i < Dim; ++i)
    result += a[i] * b[i];

  return result;
}



template<std::size_t Dim>
vector_n<Dim> normalize(const vector_n<Dim>& v)
{
  vector_n<Dim> result;
  scalar length_inv = 1. / std::sqrt(dot(v,v));
  for(std::size_t i = 0; i < Dim; ++i)
    result[i] = length_inv * v[i];
  return result;
}

inline static
vector3 cross(const vector3& a, const vector3& b)
{
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

inline scalar square(scalar x)
{
  return x*x;
}


static
void matrix_create_rotation_matrix(matrix3x3* m, vector3 axis, scalar alpha)
{
  scalar a1 = VECTOR3_X(axis);
  scalar a2 = VECTOR3_Y(axis);
  scalar a3 = VECTOR3_Z(axis);
  scalar a1a2 = a1 * a2;
  scalar a1a3 = a1 * a3;
  scalar a2a3 = a2 * a3;

#ifndef __OPENCL_VERSION__
  scalar cos_alpha = std::cos(alpha);
  scalar sin_alpha = std::sin(alpha);
#else
  scalar cos_alpha = cos(alpha);
  scalar sin_alpha = sin(alpha);
#endif
  scalar flipped_cos_alpha = 1.0 - cos_alpha;

  VECTOR3_X(m->row0) = a1*a1*flipped_cos_alpha + cos_alpha;
  VECTOR3_Y(m->row0) = a1a2 *flipped_cos_alpha - a3*sin_alpha;
  VECTOR3_Z(m->row0) = a1a3 *flipped_cos_alpha + a2*sin_alpha;

  VECTOR3_X(m->row1) = a1a2 *flipped_cos_alpha + a3*sin_alpha;
  VECTOR3_Y(m->row1) = a2*a2*flipped_cos_alpha + cos_alpha;
  VECTOR3_Z(m->row1) = a2a3 *flipped_cos_alpha - a1*sin_alpha;

  VECTOR3_X(m->row2) = a1a3 *flipped_cos_alpha - a2*sin_alpha;
  VECTOR3_Y(m->row2) = a2a3 *flipped_cos_alpha + a1*sin_alpha;
  VECTOR3_Z(m->row2) = a3*a3*flipped_cos_alpha + cos_alpha;
}

inline static
vector3 matrix_vector_mult(const matrix3x3& m, const vector3& v)
{
  vector3 result;
  VECTOR3_X(result) = dot(m.row0, v);
  VECTOR3_Y(result) = dot(m.row1, v);
  VECTOR3_Z(result) = dot(m.row2, v);
  return result;
}

inline static
matrix3x3 matrix_matrix_mult(const matrix3x3& A, const matrix3x3& B)
{
  matrix3x3 result;
  vector3 B_column0 = VECTOR3(VECTOR3_X(B.row0), VECTOR3_X(B.row1), VECTOR3_X(B.row2));
  vector3 B_column1 = VECTOR3(VECTOR3_Y(B.row0), VECTOR3_Y(B.row1), VECTOR3_Y(B.row2));
  vector3 B_column2 = VECTOR3(VECTOR3_Z(B.row0), VECTOR3_Z(B.row1), VECTOR3_Z(B.row2));

  VECTOR3_X(result.row0) = dot(A.row0, B_column0);
  VECTOR3_Y(result.row0) = dot(A.row0, B_column1);
  VECTOR3_Z(result.row0) = dot(A.row0, B_column2);

  VECTOR3_X(result.row1) = dot(A.row1, B_column0);
  VECTOR3_Y(result.row1) = dot(A.row1, B_column1);
  VECTOR3_Z(result.row1) = dot(A.row1, B_column2);

  VECTOR3_X(result.row2) = dot(A.row2, B_column0);
  VECTOR3_Y(result.row2) = dot(A.row2, B_column1);
  VECTOR3_Z(result.row2) = dot(A.row2, B_column2);

  return result;
}

namespace geometry {

inline bool is_within_range(scalar x, scalar min, scalar max)
{
  return min <= x && x <= max;
}

inline bool is_point_within_rectangle(const vector2& point,
                                      const vector2& rectangle_min_corner,
                                      const vector2& rectangle_max_corner)
{
  return is_within_range(point[0], rectangle_min_corner[0], rectangle_max_corner[0])
      && is_within_range(point[1], rectangle_min_corner[1], rectangle_max_corner[1]);
}

inline
bool rectangle_intersection(const vector2& min_a, const vector2& max_a,
                            const vector2& min_b, const vector2& max_b)
{
  return min_a[0] < max_b[0]
      && min_b[0] < max_a[0]
      && min_a[1] < max_b[1]
      && min_b[1] < max_a[1];
}

}
}

template<std::size_t Dim>
math::vector_n<Dim>& operator+=(math::vector_n<Dim>& a, const math::vector_n<Dim>& b)
{
  for(std::size_t i = 0; i < Dim; ++i)
    a[i] += b[i];

  return a;
}

template<std::size_t Dim>
math::vector_n<Dim>& operator-=(math::vector_n<Dim>& a, const math::vector_n<Dim>& b)
{
  for(std::size_t i = 0; i < Dim; ++i)
    a[i] -= b[i];

  return a;
}

template<std::size_t Dim>
math::vector_n<Dim>& operator*=(math::vector_n<Dim>& a, const math::scalar b)
{
  for(std::size_t i = 0; i < Dim; ++i)
    a[i] *= b;

  return a;
}

template<std::size_t Dim>
math::vector_n<Dim> operator+(const math::vector_n<Dim>& a, const math::vector_n<Dim>& b)
{
  math::vector_n<Dim> result = a;
  result += b;
  return result;
}

template<std::size_t Dim>
math::vector_n<Dim> operator-(const math::vector_n<Dim>& a, const math::vector_n<Dim>& b)
{
  math::vector_n<Dim> result = a;
  result -= b;
  return result;
}

template<std::size_t Dim>
math::vector_n<Dim> operator*(const math::vector_n<Dim>& a, const math::scalar b)
{
  math::vector_n<Dim> result = a;
  result *= b;
  return result;
}

template<std::size_t Dim>
math::vector_n<Dim> operator*(const math::scalar b, const math::vector_n<Dim>& a)
{
  return a * b;
}

namespace math {
template<std::size_t Dim>
inline scalar distance2(const vector_n<Dim>& a, const vector_n<Dim>& b)
{
  vector_n<Dim> R = a;
  R -= b;

  return dot(R,R);
}

} // math


}

#endif
