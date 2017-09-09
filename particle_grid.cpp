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

#include "particle_grid.hpp"

namespace illcrawl {

particle_grid::particle_grid(const qcl::device_context_ptr& ctx,
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

device_vector3 particle_grid::get_grid_min_corner() const
{
  return _grid_min_corner;
}

cl_int3 particle_grid::get_num_grid_cells() const
{
  return _num_cells;
}

device_vector3 particle_grid::get_grid_cell_sizes() const
{
  return _cell_sizes;
}

const cl::Buffer& particle_grid::get_grid_cells_buffer() const
{
  return _cells_buffer;
}

const cl::Buffer& particle_grid::get_particle_buffer() const
{
  return _particles_buffer;
}

const cl::Event& particle_grid::get_grid_ready_event() const
{
  return _grid_ready;
}

void particle_grid::build_tiles(std::size_t num_particles)
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



}
