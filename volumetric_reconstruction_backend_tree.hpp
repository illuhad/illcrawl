
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


#ifndef VOLUMETRIC_RECONSTRUCTION_BACKEND_TREE_HPP
#define VOLUMETRIC_RECONSTRUCTION_BACKEND_TREE_HPP

#include <vector>
#include <string>
#include <memory>

#include "reconstruction_backend.hpp"
#include "qcl.hpp"
#include "cl_types.hpp"
#include "async_io.hpp"
#include "interpolation_tree.hpp"

namespace illcrawl {
namespace reconstruction_backends {

class tree : public reconstruction_backend
{
public:
  virtual ~tree(){}

  tree(const qcl::device_context_ptr& ctx,
       math::scalar opening_angle);

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
  cl::Buffer _evaluation_points;
  cl::Buffer _evaluation_value_sums;
  cl::Buffer _evaluation_weight_sums;
  cl::Buffer _result;

  std::size_t _num_evaluation_points;

  qcl::device_context_ptr _ctx;
  std::size_t _blocksize;

  math::scalar _opening_angle;

  static constexpr std::size_t local_size = 256;

  std::unique_ptr<sparse_interpolation_tree> _tree;

};

}
}

#endif
