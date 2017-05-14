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


#ifndef PARTICLE_TILES
#define PARTICLE_TILES

#include <vector>
#include <cmath>
#include <array>
#include <algorithm>

#include "math.hpp"
#include "qcl.hpp"
#include "grid.hpp"
#include "multi_array.hpp"


namespace illcrawl {

/// Sorts a collection of particles into a grid and
/// transfers the grid to the GPU
class particle_tile_grid
{
public:
  using particle = cl_float4;
  using scalar = cl_float;

  static constexpr std::size_t target_num_particles_per_tile = 8;

  /// Data stored in the tiles:
  /// tile_descriptor.s[0] -- Number of particles in this tile
  /// tile_descriptor.s[1] -- Only used temporarily during the
  ///    sorting to store the number of already placed particles
  /// tile_descriptor.s[2] -- The offset where the first particle
  ///    belonging to this file can be found
  /// tile_descriptor.s[3] -- The maximum smoothing length of particles
  ///    in this tile
  using tile_descriptor = cl_float4;

  /// particle Layout:
  /// particle.s[0] -- x coordinate
  /// particle.s[1] -- y coordinate
  /// particle.s[2] -- z coordiante
  /// particle.s[3] -- quantity to reconstruct
  particle_tile_grid(const qcl::device_context_ptr& ctx,
                 const std::vector<particle>& particles,
                 const std::vector<scalar>& smoothing_lengths)
    : _ctx{ctx},
      _max_smoothing_length{0.0f},
      _sorted_particles(particles.size()),
      _transfers_completed(2)
  {
    assert(ctx != nullptr);
    assert(particles.size() == smoothing_lengths.size());

    math::vector3 min_particle_coordinates = {{-1., -1., -1.}};
    math::vector3 max_particle_coordinates = {{ 1.,  1.,  1.}};

    _ctx->create_input_buffer<particle>(_particles_buffer,
                                        particles.size());

    if(particles.size() > 0)
    {
      for(std::size_t j = 0; j < 3; ++j)
      {
        min_particle_coordinates[j] =
            static_cast<math::scalar>(particles[0].s[j]);

        max_particle_coordinates[j] =
            static_cast<math::scalar>(particles[0].s[j]);
      }
    }

    for(std::size_t i = 0; i < particles.size(); ++i)
    {
      particle current_particle = particles[i];
      for(std::size_t j = 0; j < 3; ++j)
      {
        if(static_cast<math::scalar>(current_particle.s[j])
           < min_particle_coordinates[j])
        {
          min_particle_coordinates[j] =
              static_cast<math::scalar>(current_particle.s[j]);
        }
        if(static_cast<math::scalar>(current_particle.s[j])
           > max_particle_coordinates[j])
        {
          max_particle_coordinates[j] =
              static_cast<math::scalar>(current_particle.s[j]);
        }
      }

      scalar smoothing_length =  smoothing_lengths[i];

      if(smoothing_length > _max_smoothing_length)
        _max_smoothing_length = smoothing_length;
    }

    math::scalar grid_volume = 1.0;
    for(std::size_t i = 0; i < 3; ++i)
    {
      // Make sure the grid boundaries are not
      // exactly on the particles
      min_particle_coordinates[i] -= 0.1;
      max_particle_coordinates[i] += 0.1;

      assert(max_particle_coordinates[i] > min_particle_coordinates[i]);
      grid_volume *= (max_particle_coordinates[i] - min_particle_coordinates[i]);
    }

    math::scalar tile_volume = static_cast<math::scalar>(target_num_particles_per_tile) /
                               static_cast<math::scalar>(particles.size()) * grid_volume;
    math::scalar tile_width = std::cbrt(tile_volume);

    std::array<std::size_t, 3> num_tiles;
    for(std::size_t i = 0; i < 3; ++i)
    {
      math::scalar delta = max_particle_coordinates[i] - min_particle_coordinates[i];
      num_tiles[i] = static_cast<std::size_t>(std::ceil(delta / tile_width));
    }

    this->_tiles_grid = util::grid_coordinate_translator<3>{
        0.5 * (min_particle_coordinates + max_particle_coordinates), // grid center
        max_particle_coordinates - min_particle_coordinates, // grid extent
        num_tiles // num grid cells
    };

    std::cout << "Using "
              << num_tiles[0] << "/"
              << num_tiles[1] << "/"
              << num_tiles[2] << " tiles."<<std::endl;
    this->_tiles = util::multi_array<tile_descriptor>{
        num_tiles[0],
        num_tiles[1],
        num_tiles[2]
    };

    std::fill(_tiles.begin(), _tiles.end(), empty_tile);

    count_particles_per_tile(particles);
    //for(std::size_t i = 0; i < _tiles.get_num_elements(); ++i)
    //  std::cout << _tiles.data()[i].s[0] << std::endl;

    calculate_particle_offsets();
    sort_into_tiles(particles, smoothing_lengths);


    _ctx->create_input_buffer<tile_descriptor>(_tiles_buffer,
                                               _tiles.get_num_elements());

    _ctx->memcpy_h2d_async(_tiles_buffer,
                           _tiles.data(),
                           _tiles.get_num_elements(),
                           &(_transfers_completed[0]));

    _ctx->memcpy_h2d_async(_particles_buffer,
                           _sorted_particles.data(),
                           _sorted_particles.size(),
                           &(_transfers_completed[1]));
  }



  std::array<std::size_t, 3> get_num_tiles() const
  {
    return _tiles_grid.get_num_cells();
  }

  const std::vector<cl::Event>* get_data_transferred_events() const
  {
    return &_transfers_completed;
  }

  const cl::Buffer& get_tiles_buffer() const
  {
    return _tiles_buffer;
  }

  const cl::Buffer& get_sorted_particles_buffer() const
  {
    return _particles_buffer;
  }

  math::vector3 get_tiles_min_corner() const
  {
    return _tiles_grid.get_grid_min_corner();
  }

  math::vector3 get_tile_sizes() const
  {
    return _tiles_grid.get_cell_sizes();
  }

  scalar get_maximum_smoothing_length() const
  {
    return _max_smoothing_length;
  }

private:
  const tile_descriptor empty_tile = {{0.0f, 0.0f, 0.0f, 0.0f}};


  void count_particles_per_tile(const std::vector<particle>& particles)
  {
    // Count the number of particles for each tile
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
      particle current_particle = particles[i];

      math::vector3 particle_coordinates;
      for(std::size_t j = 0; j < 3; ++j)
        particle_coordinates[j] = static_cast<math::scalar>(current_particle.s[j]);

      auto grid_idx = _tiles_grid(particle_coordinates);
      assert(_tiles_grid.is_within_bounds(grid_idx));

      auto grid_uidx = _tiles_grid.unsigned_grid_index(grid_idx);
      _tiles[grid_uidx].s[0] += 1.0f;
    }
  }

  void calculate_particle_offsets()
  {
    // Set offsets for particles
    for(std::size_t i = 0; i < _tiles.get_num_elements(); ++i)
    {
      scalar previous_offset = 0.0;
      scalar previous_num_particles = 0.0;
      if(i > 0)
      {
        previous_offset        = _tiles.data()[i-1].s[2];
        previous_num_particles = _tiles.data()[i-1].s[0];
      }
      _tiles.data()[i].s[2] = previous_offset + previous_num_particles;
    }
  }

  void sort_into_tiles(const std::vector<particle>& particles,
                       const std::vector<scalar>& smoothing_lengths)
  {
    // Sort particles
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
      particle current_particle = particles[i];

      math::vector3 particle_coordinates;
      for(std::size_t j = 0; j < 3; ++j)
        particle_coordinates[j] =
            static_cast<math::scalar>(current_particle.s[j]);

      auto grid_idx = _tiles_grid(particle_coordinates);

      assert(_tiles_grid.is_within_bounds(grid_idx));

      auto grid_uidx = _tiles_grid.unsigned_grid_index(grid_idx);

      std::size_t already_placed_particles =
          static_cast<std::size_t>(_tiles[grid_uidx].s[1]);

      std::size_t offset =
          static_cast<std::size_t>(_tiles[grid_uidx].s[2]);

      assert(already_placed_particles + offset < _sorted_particles.size());
      _sorted_particles[already_placed_particles + offset] = current_particle;

      // Increase number of already placed particles
      _tiles[grid_uidx].s[1] += 1;

      // Update maximum smoothing length of tile
      if(_tiles[grid_uidx].s[3] < smoothing_lengths[i])
        _tiles[grid_uidx].s[3] = smoothing_lengths[i];
    }
  }

  qcl::device_context_ptr _ctx;

  scalar _max_smoothing_length;

  util::grid_coordinate_translator<3> _tiles_grid;
  util::multi_array<tile_descriptor> _tiles;
  std::vector<particle> _sorted_particles;

  cl::Buffer _tiles_buffer;
  cl::Buffer _particles_buffer;

  std::vector<cl::Event> _transfers_completed;

};

}

#endif
