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

#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <cassert>
#include <random>
#include <limits>
#include <cmath>
#include "math.hpp"

namespace illcrawl {
namespace random {

class random_number_generator
{
public:
  random_number_generator(std::size_t seed)
    : _random_engine{seed}
  {}

  random_number_generator()
    : _random_engine{_rd()}
  {}

  template<class Real_type>
  Real_type uniform_real(Real_type a = 0.0, Real_type b = 1.0) const
  {
    return a + (b-a) * static_cast<Real_type>(_real_distribution(_random_engine));
  }

  template<class Int_type>
  Int_type uniform_int(Int_type a = 0,
                       Int_type b = std::numeric_limits<Int_type>::max()) const
  {
    assert(b > a);
    return a + static_cast<Int_type>(_int_distribution(_random_engine)) % (b - a);
  }

private:
  std::random_device _rd;
  mutable std::mt19937 _random_engine;

  mutable std::uniform_real_distribution<double> _real_distribution;
  mutable std::uniform_int_distribution<std::size_t> _int_distribution;
};

namespace sampler {
/// Generates a random, uniformly sampled, point on the
/// surface of the unit sphere
class uniform_sphere_surface
{
public:

  inline
  math::vector3 operator()(const random_number_generator& rng) const
  {
    math::scalar x1, x2;
    math::scalar r;
    do
    {
      x1 = rng.uniform_real<math::scalar>(-1., 1.);
      x2 = rng.uniform_real<math::scalar>(-1., 1.);
      r = x1*x1 + x2*x2;
    } while(r > 1.);

    math::scalar sqrt_term = std::sqrt(1. - r);

    return {{2 * x1 * sqrt_term,
             2 * x2 * sqrt_term,
             1 - 2 * r}};
  }
};

/// Uniformly samples the volume of the unit sphere
class uniform_sphere_volume
{
public:

  inline
  math::vector3 operator()(const random_number_generator& rng) const
  {
    math::scalar x1,x2,x3;
    math::scalar r;
    do
    {
      x1 = rng.uniform_real<math::scalar>(-1., 1.);
      x2 = rng.uniform_real<math::scalar>(-1., 1.);
      x3 = rng.uniform_real<math::scalar>(-1., 1.);

      r = x1*x1 + x2*x2 + x3*x3;
    } while(r > 1.);

    return {{x1, x2, x3}};
  }
};

/// Uniformly sample a spherical shell
class uniform_spherical_shell
{
public:
  uniform_spherical_shell(math::scalar r_min, math::scalar r_max)
    : _r_min{r_min},
      _r_max{r_max},
      _u_min{r_min * r_min * r_min},
      _u_max{r_max * r_max * r_max}
  {}

  inline
  math::vector3 operator()(const random_number_generator& rng) const
  {
    uniform_sphere_surface surface_sampler;
    // First sample point on surface
    math::vector3 sample = surface_sampler(rng);
    // Then correct radius
    math::scalar u = rng.uniform_real<math::scalar>(_u_min, _u_max);
    return std::cbrt(u) * sample;
  }

private:

  math::scalar _r_min;
  math::scalar _r_max;

  math::scalar _u_min;
  math::scalar _u_max;
};

} // sampler
} // random
} // illcrawl

#endif
