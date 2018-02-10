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

#include <cassert>

#include "smoothing_particle_grid.hpp"

#include <boost/compute/algorithm/max_element.hpp>

namespace illcrawl {

smoothing_particle_grid::smoothing_particle_grid(
                        const qcl::device_context_ptr& ctx,
                        const std::vector<particle>& particles,
                        const cl::Buffer& smoothing_lengths,
                        std::size_t target_num_particles_per_cell)
  : particle_grid{ctx, particles, target_num_particles_per_cell}
{
  this->_sorted_smoothing_lengths = smoothing_lengths;
  this->sort_scalars_into_cells(_sorted_smoothing_lengths);

  // Determine maximum smoothing length of the give particle set
  boost::compute::buffer_iterator<device_scalar> max_element_iterator =
      boost::compute::max_element(
        qcl::create_buffer_iterator<device_scalar>(_sorted_smoothing_lengths, 0),
        qcl::create_buffer_iterator<device_scalar>(_sorted_smoothing_lengths,
                                                   this->get_num_particles()),
        this->get_boost_queue());

  assert(max_element_iterator.get_index() < this->get_num_particles());

  _maximum_smoothing_length = max_element_iterator.read(this->get_boost_queue());

  // Determine maximum smoothing length for each cell
  this->determine_max_values_for_cells(_sorted_smoothing_lengths,
                                       _maximum_smoothing_lengths_for_cells);
}

const cl::Buffer&
smoothing_particle_grid::get_sorted_smoothing_lengths() const
{
  return this->_sorted_smoothing_lengths;
}

const cl::Buffer&
smoothing_particle_grid::get_max_smoothing_length_per_cell() const
{
  return this->_maximum_smoothing_lengths_for_cells;
}

device_scalar
smoothing_particle_grid::get_overall_max_smoothing_length() const
{
  return this->_maximum_smoothing_length;
}

}
