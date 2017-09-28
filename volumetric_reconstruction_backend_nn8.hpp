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

#ifndef VOLUMETRIC_RECONSTRUCTION_BACKEND_NN8
#define VOLUMETRIC_RECONSTRUCTION_BACKEND_NN8

#include <vector>
#include <string>

#include "reconstruction_backend.hpp"
#include "qcl.hpp"
#include "cl_types.hpp"
#include "async_io.hpp"
#include "particle_grid.hpp"

namespace illcrawl {
namespace reconstruction_backends {

class nn8 : public reconstruction_backend
{
public:
  virtual ~nn8(){}

  using nearest_neighbor_list = device_vector8;

  nn8(const qcl::device_context_ptr& ctx);

  virtual std::vector<H5::DataSet> get_required_additional_datasets() const override;

  virtual std::string get_backend_name() const override;

  virtual void init_backend(std::size_t blocksize) override;

  virtual void setup_particles(const std::vector<particle>& particles,
                               const std::vector<cl::Buffer>& additional_dataset) override;

  virtual void setup_evaluation_points(const cl::Buffer& evaluation_points,
                                       std::size_t num_points) override;

  virtual void run() override;

  virtual const cl::Buffer& retrieve_results() override;

private:

  void launch_reconstruction(cl::Event* kernel_finished_event = nullptr);
  void finalize_results(cl::Event* finalization_done = nullptr);

  qcl::device_context_ptr _ctx;
  std::size_t _blocksize;

  std::unique_ptr<particle_grid> _grid;

  std::size_t _num_evaluation_points;
  cl::Buffer _result;

  cl::Buffer _weights_buffer;
  cl::Buffer _values_buffer;
  cl::Buffer _evaluation_points;

  /// whether the kernel launch is the first for a given set
  /// of evaluation points
  bool _is_first_run = true;

  static constexpr std::size_t local_size = 256;
};

}
}

#endif
