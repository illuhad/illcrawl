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
                const std::vector<particle>& particles)
    : _ctx{ctx}, _boost_queue{ctx->get_command_queue().get()}
  {
    assert(ctx != nullptr);

    _ctx->create_buffer<particle>(_particles_buffer,
                                  CL_MEM_READ_WRITE,
                                  particles.size());

    cl::Event particles_transferred;
    _ctx->memcpy_h2d_async(_particles_buffer,
                           particles.data(), particles.size(),
                           &particles_transferred);

    // Determine grid boundaries
    math::vector3 min_particle_coordinates = {{-1., -1., -1.}};
    math::vector3 max_particle_coordinates = {{ 1.,  1.,  1.}};
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

    for(const particle& p : particles)
    {
      for(std::size_t i = 0; i < 3; ++i)
      {
        if(static_cast<math::scalar>(p.s[i]) < min_particle_coordinates[i])
          min_particle_coordinates[i] = static_cast<math::scalar>(p.s[i]);
        if(static_cast<math::scalar>(p.s[i]) > max_particle_coordinates[i])
          max_particle_coordinates[i] = static_cast<math::scalar>(p.s[i]);
      }
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

    for(std::size_t i = 0; i < 3; ++i)
    {
      _num_cells.s[i] = static_cast<cl_int>(
            std::ceil((max_particle_coordinates[i] - min_particle_coordinates[i])/tile_width));
      _grid_min_corner.s[i] = static_cast<device_scalar>(min_particle_coordinates[i]);
      _cell_sizes.s[i] = static_cast<device_scalar>(tile_width);
    }

    _total_num_cells = static_cast<std::size_t>(_num_cells.s[0]) *
                       static_cast<std::size_t>(_num_cells.s[1]) *
                       static_cast<std::size_t>(_num_cells.s[2]);

    _ctx->create_buffer<cl_ulong>(_grid_cell_keys_buffer,
                                  CL_MEM_READ_WRITE,
                                  particles.size());
    _ctx->create_buffer<cell_descriptor>(_cells_buffer,
                                         CL_MEM_READ_WRITE,
                                         _total_num_cells);

    cell_descriptor empty_cell{0,0};
    _ctx->get_command_queue().enqueueFillBuffer(_cells_buffer, empty_cell, 0,
                                                sizeof(cell_descriptor) * _total_num_cells);


    particles_transferred.wait();

    build_tiles(particles.size());
  }

  device_vector3 get_grid_min_corner() const
  {
    return _grid_min_corner;
  }

  cl_int3 get_num_grid_cells() const
  {
    return _num_cells;
  }

  device_vector3 get_grid_cell_sizes() const
  {
    return _cell_sizes;
  }

  const cl::Buffer& get_grid_cells_buffer() const
  {
    return _cells_buffer;
  }

  const cl::Buffer& get_particle_buffer() const
  {
    return _particles_buffer;
  }

  const cl::Event& get_grid_ready_event() const
  {
    return _grid_ready;
  }
private:
  void build_tiles(std::size_t num_particles)
  {
    // Generate keys (grid cell indices) for all particles

    qcl::kernel_ptr key_generation_kernel = _ctx->get_kernel("grid3d_generate_sort_keys");

    qcl::kernel_argument_list key_creation_args{key_generation_kernel};
    key_creation_args.push(_particles_buffer);
    key_creation_args.push(_grid_cell_keys_buffer);
    key_creation_args.push(_num_cells);
    key_creation_args.push(_grid_min_corner);
    key_creation_args.push(_cell_sizes);
    key_creation_args.push(static_cast<cl_int>(num_particles));

    cl::Event keys_generated;
    cl_int err = 0;
    err = _ctx->get_command_queue().enqueueNDRangeKernel(*key_generation_kernel,
                                                   cl::NullRange,
                                                   cl::NDRange{math::make_multiple_of(local_size, num_particles)},
                                                   cl::NDRange{local_size},
                                                   nullptr, &keys_generated);
    qcl::check_cl_error(err, "Could not enqueue grid cell key generation kernel!");
    err = keys_generated.wait();
    qcl::check_cl_error(err, "Error during the execution of the grid cell key generation kernel");

    // Sort particles by grid cell indices
    boost::compute::sort_by_key(qcl::create_buffer_iterator<cl_ulong>(_grid_cell_keys_buffer, 0),
                                qcl::create_buffer_iterator<cl_ulong>(_grid_cell_keys_buffer, num_particles),
                                qcl::create_buffer_iterator<boost_particle>(_particles_buffer, 0),
                                boost::compute::less<cl_ulong>(),
                                _boost_queue);

    // Now we can actually create the grid cells!

    qcl::kernel_ptr cell_begin_kernel = _ctx->get_kernel("grid3d_determine_cells_begin");
    qcl::kernel_argument_list cell_begin_args{cell_begin_kernel};
    cell_begin_args.push(_particles_buffer);
    cell_begin_args.push(static_cast<cl_int>(num_particles));
    cell_begin_args.push(_cells_buffer);
    cell_begin_args.push(_grid_min_corner);
    cell_begin_args.push(_cell_sizes);
    cell_begin_args.push(_num_cells);

    err = _ctx->get_command_queue().enqueueNDRangeKernel(*cell_begin_kernel,
                                                         cl::NullRange,
                                                         cl::NDRange{math::make_multiple_of(local_size, num_particles)},
                                                         cl::NDRange{local_size},
                                                         nullptr,
                                                         &_grid_ready);
    qcl::check_cl_error(err, "Could not enqueue cell begin kernel");

    qcl::kernel_ptr cell_end_kernel = _ctx->get_kernel("grid3d_determine_cells_end");
    qcl::kernel_argument_list cell_end_args{cell_end_kernel};
    cell_end_args.push(_particles_buffer);
    cell_end_args.push(static_cast<cl_int>(num_particles));
    cell_end_args.push(_cells_buffer);
    cell_end_args.push(_grid_min_corner);
    cell_end_args.push(_cell_sizes);
    cell_end_args.push(_num_cells);

    _grid_ready.wait();
    err = _ctx->get_command_queue().enqueueNDRangeKernel(*cell_end_kernel,
                                                         cl::NullRange,
                                                         cl::NDRange{math::make_multiple_of(local_size, num_particles)},
                                                         cl::NDRange{local_size},
                                                         nullptr,
                                                         &_grid_ready);
    qcl::check_cl_error(err, "Could not enqueue cell end kernel");
  }

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

/// Sorts a collection of particles into a grid and
/// transfers the grid to the GPU
class particle_tile_grid
{
public:
  using particle = device_vector4;

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
                 const std::vector<device_scalar>& smoothing_lengths)
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

      device_scalar smoothing_length =  smoothing_lengths[i];

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

  device_scalar get_maximum_smoothing_length() const
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
      device_scalar previous_offset = 0.0;
      device_scalar previous_num_particles = 0.0;
      if(i > 0)
      {
        previous_offset        = _tiles.data()[i-1].s[2];
        previous_num_particles = _tiles.data()[i-1].s[0];
      }
      _tiles.data()[i].s[2] = previous_offset + previous_num_particles;
    }
  }

  void sort_into_tiles(const std::vector<particle>& particles,
                       const std::vector<device_scalar>& smoothing_lengths)
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

      if(already_placed_particles + offset >= _sorted_particles.size())
      {
        std::cout << "already_places_particles: " << already_placed_particles << " offset " << offset << " size " << _sorted_particles.size() << std::endl;
        std::cout << _tiles[grid_uidx].s[1] << " " << _tiles[grid_uidx].s[2] << std::endl;
      }
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

  device_scalar _max_smoothing_length;

  util::grid_coordinate_translator<3> _tiles_grid;
  util::multi_array<tile_descriptor> _tiles;
  std::vector<particle> _sorted_particles;

  cl::Buffer _tiles_buffer;
  cl::Buffer _particles_buffer;

  std::vector<cl::Event> _transfers_completed;

};

}

#endif
