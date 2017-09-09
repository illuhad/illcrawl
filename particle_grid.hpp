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


#ifndef PARTICLE_GRID
#define PARTICLE_GRID

#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <boost/compute.hpp>

#include "cl_types.hpp"
#include "math.hpp"
#include "qcl.hpp"
#include "grid.hpp"
#include "multi_array.hpp"


namespace illcrawl {

class particle_grid
{
public:
  /// particle Layout:
  /// particle.s[0] -- x coordinate
  /// particle.s[1] -- y coordinate
  /// particle.s[2] -- z coordiante
  /// particle.s[3] -- quantity to reconstruct
  using particle = device_vector4;
  using boost_particle = boost_device_vector4;

  static constexpr std::size_t target_num_particles_per_tile = 8;

  /// Data stored in the tiles:
  /// tile_descriptor.s[0] -- Index of first particle in tile
  /// tile_descriptor.s[1] -- Index of first particle not in the tile
  /// anymore.
  using cell_descriptor = cl_int2;
  using boost_cell_descriptor = boost::compute::int2_;

  particle_grid(const qcl::device_context_ptr& ctx,
                const std::vector<particle>& particles);

  device_vector3 get_grid_min_corner() const;

  cl_int3 get_num_grid_cells() const;

  device_vector3 get_grid_cell_sizes() const;

  const cl::Buffer& get_grid_cells_buffer() const;

  const cl::Buffer& get_particle_buffer() const;

  const cl::Event& get_grid_ready_event() const;
private:
  void build_tiles(std::size_t num_particles);

  static constexpr std::size_t local_size = 512;

  cl::Buffer _particles_buffer;
  cl::Buffer _grid_cell_keys_buffer;
  cl::Buffer _cells_buffer;

  qcl::device_context_ptr _ctx;

  std::size_t _total_num_cells;
  cl_int3 _num_cells;
  device_vector3 _grid_min_corner;
  device_vector3 _cell_sizes;

  cl::Event _grid_ready;

  boost::compute::command_queue _boost_queue;
};


}

#endif
