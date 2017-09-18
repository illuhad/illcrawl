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

#ifndef SMOOTHING_PARTICLE_GRID
#define SMOOTHING_PARTICLE_GRID

#include "particle_grid.hpp"

namespace illcrawl {

class smoothing_particle_grid : public particle_grid
{
public:
  smoothing_particle_grid(const qcl::device_context_ptr& ctx,
                          const std::vector<particle>& particles,
                          const cl::Buffer& smoothing_lengths);

  const cl::Buffer& get_sorted_smoothing_lengths() const;
  const cl::Buffer& get_max_smoothing_length_per_cell() const;
  device_scalar get_overall_max_smoothing_length() const;

private:
  cl::Buffer _sorted_smoothing_lengths;
  cl::Buffer _maximum_smoothing_lengths_for_cells;

  device_scalar _maximum_smoothing_length = 0.0f;
};

}

#endif
