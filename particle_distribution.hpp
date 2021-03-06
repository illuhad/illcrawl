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


#ifndef PARTICLE_DISTRIBUTION_HPP
#define PARTICLE_DISTRIBUTION_HPP

#include <array>
#include <limits>

#include "async_io.hpp"
#include "math.hpp"


namespace illcrawl {

class particle_distribution
{
public:
  particle_distribution(const H5::DataSet& coordinates,
                        const math::vector3& periodic_wraparound_size);

  const math::vector3& get_extent_center() const;

  const math::vector3& get_distribution_size() const;

  const math::vector3& get_mean_particle_position() const;
private:
  math::vector3 _distribution_center;
  math::vector3 _distribution_size;
  math::vector3 _mean_particle_position;
};

}

#endif
