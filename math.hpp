#ifndef MATH_HPP
#define MATH_HPP

#include <array>
#include <cmath>

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




}

#endif
