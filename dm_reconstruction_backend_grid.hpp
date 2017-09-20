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

#ifndef DM_RECONSTRUCTION_BACKEND_GRID
#define DM_RECONSTRUCTION_BACKEND_GRID

#include "qcl.hpp"
#include "reconstruction_backend.hpp"
#include "smoothing_particle_grid.hpp"

namespace illcrawl {
namespace reconstruction_backends {
namespace dm {

class grid : virtual public reconstruction_backend
{
public:
  grid(const qcl::device_context_ptr& ctx,
       const H5::DataSet& smoothing_lengths,
       const std::string& reconstruction_kernel_variant =
                  "dm_reconstruction_grid_smoothing",
       std::size_t target_num_particles_per_cell = 64);

  virtual std::vector<H5::DataSet> get_required_additional_datasets() const override;

  virtual const cl::Buffer& retrieve_results() override;

  virtual std::string get_backend_name() const override;

  virtual void init_backend(std::size_t blocksize) override;

  virtual void setup_particles(const std::vector<particle>& particles,
                               const std::vector<cl::Buffer>& additional_dataset) override;

  virtual void setup_evaluation_points(const cl::Buffer& evaluation_points,
                                       std::size_t num_points) override;

  virtual void run() override;

  virtual ~grid(){}

private:
  qcl::device_context_ptr _ctx;
  H5::DataSet _smoothing_lengths;

  std::size_t _blocksize;

  cl::Buffer _result_buffer;
  cl::Buffer _evaluation_points_buffer;

  std::size_t _num_evaluation_points;
  std::size_t _num_particles;

  std::unique_ptr<smoothing_particle_grid> _grid;

  device_scalar _maximum_smoothing_length = 0.0f;

  std::string _reconstruction_kernel_name;
  std::size_t _target_num_particles_per_cell;

  static constexpr std::size_t _local_size = 512;
};

}
}
}

#endif
