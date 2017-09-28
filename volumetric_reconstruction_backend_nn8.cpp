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

#include "volumetric_reconstruction_backend_nn8.hpp"

namespace illcrawl {
namespace reconstruction_backends {

nn8::nn8(const qcl::device_context_ptr& ctx)
  : _ctx{ctx},
    _blocksize{0},
    _num_evaluation_points{0}
{}

std::vector<H5::DataSet> nn8::get_required_additional_datasets() const
{
  return std::vector<H5::DataSet>();
}

std::string nn8::get_backend_name() const
{
  return "volumetric/nn8";
}

void nn8::init_backend(std::size_t blocksize)
{
  this->_blocksize = blocksize;
}

void nn8::setup_particles(const std::vector<particle>& particles,
                          const std::vector<cl::Buffer>& additional_dataset)
{
  // First release memory before allocating new memory
  _grid = nullptr;
  _grid = std::unique_ptr<particle_grid>{
      new particle_grid{_ctx, particles}
  };
}

void nn8::setup_evaluation_points(const cl::Buffer& evaluation_points,
                                  std::size_t num_points)
{
  this->_num_evaluation_points = num_points;
  this->_evaluation_points = evaluation_points;
  this->_is_first_run = true;
  _ctx->create_buffer<nearest_neighbor_list>(_weights_buffer, num_points);
  _ctx->create_buffer<nearest_neighbor_list>(_values_buffer, num_points);
  _ctx->create_buffer<device_scalar>(_result, num_points);
}

void nn8::run()
{
  assert(_grid != nullptr);
  assert(_num_evaluation_points > 0);

  this->launch_reconstruction();
  this->_is_first_run = false;
}

const cl::Buffer& nn8::retrieve_results()
{
  this->finalize_results();
  cl_int err = this->_ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while waiting for the finalization kernel to finish");

  return _result;
}

void nn8::launch_reconstruction(cl::Event* kernel_finished_event)
{
  auto num_cells = _grid->get_num_grid_cells();
  auto grid_min_corner = _grid->get_grid_min_corner();
  auto cell_sizes = _grid->get_grid_cell_sizes();

  qcl::kernel_ptr kernel = _ctx->get_kernel("volumetric_nn8_reconstruction");

  qcl::kernel_argument_list args{kernel};
  args.push(static_cast<cl_int>(_is_first_run));
  args.push(_grid->get_grid_cells_buffer());
  args.push(num_cells);
  args.push(grid_min_corner);
  args.push(cell_sizes);

  args.push(_grid->get_particle_buffer());

  args.push(static_cast<cl_int>(_num_evaluation_points));
  args.push(_evaluation_points);
  args.push(_weights_buffer);
  args.push(_values_buffer);

  std::vector<cl::Event> grid_ready = {{_grid->get_grid_ready_event()}};

  cl_int err = _ctx->enqueue_ndrange_kernel(kernel,
                                            cl::NDRange{_num_evaluation_points},
                                            cl::NDRange{local_size},
                                            kernel_finished_event,
                                            cl::NullRange,
                                            &grid_ready,
                                            0);

  qcl::check_cl_error(err, "Could not enqueue volumetric_nn8_reconstruction kernel");
}

void nn8::finalize_results(cl::Event* finalization_done)
{
  qcl::kernel_ptr finalization_kernel = _ctx->get_kernel("finalize_volumetric_nn8_reconstruction");

  qcl::kernel_argument_list finalization_args{finalization_kernel};
  finalization_args.push(static_cast<cl_int>(_num_evaluation_points));
  finalization_args.push(_weights_buffer);
  finalization_args.push(_values_buffer);
  finalization_args.push(_result);

  cl_int err = _ctx->enqueue_ndrange_kernel(finalization_kernel,
                                            cl::NDRange{_num_evaluation_points},
                                            cl::NDRange{local_size},
                                            finalization_done);

  qcl::check_cl_error(err, "Could not enqueue finalization kernel.");

}

}
}
