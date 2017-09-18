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


#include "dm_reconstruction_backend_grid.hpp"

namespace illcrawl {
namespace reconstruction_backends {
namespace dm {

grid::grid(const qcl::device_context_ptr& ctx,
           const H5::DataSet& smoothing_lengths)
  : _ctx{ctx},
    _smoothing_lengths{smoothing_lengths},
    _blocksize{0},
    _num_evaluation_points{0},
    _num_particles{0}
{}

std::vector<H5::DataSet>
grid::get_required_additional_datasets() const
{
  return std::vector<H5::DataSet>{_smoothing_lengths};
}

const cl::Buffer&
grid::retrieve_results()
{
  return _result_buffer;
}

std::string
grid::get_backend_name() const
{
  return "dm_smoothing/grid";
}

void
grid::init_backend(std::size_t blocksize)
{
  this->_blocksize = blocksize;
}

void
grid::setup_particles(const std::vector<particle>& particles,
                      const std::vector<cl::Buffer>& additional_dataset)
{
  assert(_blocksize != 0);
  assert(particles.size() <= this->_blocksize);
  assert(additional_dataset.size() == 1);

  _grid = std::move(std::unique_ptr<smoothing_particle_grid>{
      new smoothing_particle_grid{_ctx, particles, additional_dataset[0]}
   });

  _num_particles = particles.size();

}

void
grid::setup_evaluation_points(const cl::Buffer& evaluation_points,
                              std::size_t num_points)
{
  assert(num_points > 0);
  this->_evaluation_points_buffer = evaluation_points;
  this->_ctx->create_buffer<device_scalar>(this->_result_buffer, num_points);

  cl_int err = this->_ctx->get_command_queue().enqueueFillBuffer(
        this->_result_buffer,
        static_cast<device_scalar>(0.0f),
        0,
        num_points * sizeof(device_scalar));
  qcl::check_cl_error(err, "Could not enqueue buffer fill in "
                           "grid::setup_evaluation_points()");

  err = this->_ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while waiting for buffer fill to complete");

  this->_num_evaluation_points = num_points;
}

void
grid::run()
{
  qcl::kernel_ptr reconstruction_kernel =
      _ctx->get_kernel("dm_reconstruction_grid_smoothing");

  qcl::kernel_argument_list args{reconstruction_kernel};
  args.push(_grid->get_grid_cells_buffer());
  args.push(_grid->get_num_grid_cells());
  args.push(_grid->get_grid_min_corner());
  args.push(_grid->get_grid_cell_sizes());
  args.push(_grid->get_particle_buffer());

  args.push(_grid->get_sorted_smoothing_lengths());
  args.push(_grid->get_overall_max_smoothing_length());
  args.push(_grid->get_max_smoothing_length_per_cell());

  args.push(static_cast<cl_int>(this->_num_evaluation_points));
  args.push(this->_evaluation_points_buffer);
  args.push(this->_result_buffer);

  cl_int err = _ctx->enqueue_ndrange_kernel(reconstruction_kernel,
                                            cl::NDRange{_num_evaluation_points},
                                            cl::NDRange{_local_size});
  qcl::check_cl_error(err, "Could not enqueue dm_reconstruction_grid_smoothing kernel");
}

} // dm
} // reconstruction_backends
} // illcrawl
