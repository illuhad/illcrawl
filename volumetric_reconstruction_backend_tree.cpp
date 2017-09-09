
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


#include "volumetric_reconstruction_backend_tree.hpp"

namespace illcrawl {
namespace reconstruction_backends {

tree::tree(const qcl::device_context_ptr& ctx,
           math::scalar opening_angle)
  : _num_evaluation_points{0},
    _ctx{ctx},
    _blocksize{0},
    _opening_angle{opening_angle}
{
}

std::vector<H5::DataSet>
tree::get_required_additional_datasets() const
{
  return std::vector<H5::DataSet>();
}

std::string
tree::get_backend_name() const
{
  return "volumetric/tree";
}

void
tree::init_backend(std::size_t blocksize)
{
  this->_blocksize = blocksize;
  _ctx->create_buffer<device_scalar>(this->_evaluation_value_sums, blocksize);
  _ctx->create_buffer<device_scalar>(this->_evaluation_weight_sums, blocksize);
}

void
tree::setup_particles(const std::vector<particle>& particles,
                      const std::vector<cl::Buffer>& additional_dataset)
{
  _tree = std::unique_ptr<sparse_interpolation_tree>{
      new sparse_interpolation_tree{particles, _ctx}
  };
}

void
tree::setup_evaluation_points(const cl::Buffer& evaluation_points,
                              std::size_t num_points)
{
  if(num_points > _num_evaluation_points)
    _ctx->create_buffer<device_scalar>(this->_result, num_points);

  this->_tree->purge_state();

  this->_evaluation_points = evaluation_points;
  this->_num_evaluation_points = num_points;
}

void
tree::run()
{
  assert(_tree != nullptr);
  cl::Event evaluation_complete;
  this->_tree->evaluate_tree(this->_evaluation_points,
                             this->_evaluation_value_sums,
                             this->_evaluation_weight_sums,
                             this->_result,
                             this->_num_evaluation_points,
                             this->_opening_angle,
                             &evaluation_complete);
  cl_int err = evaluation_complete.wait();
  qcl::check_cl_error(err, "Error while waiting for the tree evaluation to complete");

}

const cl::Buffer&
tree::retrieve_results()
{
  return _result;
}

}
}
