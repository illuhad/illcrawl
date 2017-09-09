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

#include "random.hpp"

namespace illcrawl {
namespace random {
namespace sampler {


math::vector3
uniform_sphere_surface::operator()(const random_number_generator& rng) const
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



math::vector3
uniform_sphere_volume::operator()(const random_number_generator& rng) const
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


uniform_spherical_shell::uniform_spherical_shell(
    math::scalar r_min,
    math::scalar r_max)
  : _r_min{r_min},
    _r_max{r_max},
    _u_min{r_min * r_min * r_min},
    _u_max{r_max * r_max * r_max}
{}

math::vector3
uniform_spherical_shell::operator()(const random_number_generator& rng) const
{
  uniform_sphere_surface surface_sampler;
  // First sample point on surface
  math::vector3 sample = surface_sampler(rng);
  // Then correct radius
  math::scalar u = rng.uniform_real<math::scalar>(_u_min, _u_max);
  return std::cbrt(u) * sample;
}


}
}
}
