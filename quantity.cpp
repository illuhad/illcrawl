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

#include "quantity.hpp"

namespace illcrawl {
namespace reconstruction_quantity {



/****** Implementation of quantity_transformation *******/

quantity_transformation::quantity_transformation(
                        const qcl::device_context_ptr& ctx,
                        const quantity& q,
                        std::size_t blocksize)
  : _ctx{ctx},
    _quantity{q},
    _blocksize{blocksize},
    _num_elements{0}
{
  _input_quantities.resize(q.get_required_datasets().size());
  _input_buffers.resize(q.get_required_datasets().size());
  _transfers_complete_events.resize(q.get_required_datasets().size());

  for(std::size_t i = 0; i < _input_quantities.size(); ++i)
    _input_quantities[i] = std::vector<device_scalar>(blocksize);

  for(std::size_t i = 0; i < _input_quantities.size(); ++i)
    _ctx->create_input_buffer<device_scalar>(_input_buffers[i], blocksize);
  _ctx->create_buffer<device_scalar>(_result, CL_MEM_READ_WRITE, blocksize);

  _scaling_factors = _quantity.get_quantitiy_scaling_factors();
}

const cl::Buffer&
quantity_transformation::get_result_buffer() const
{
  return _result;
}

void
quantity_transformation::retrieve_results(cl::Event* evt,
                                          std::vector<device_scalar>& out) const
{
  out.resize(get_num_elements());
  _ctx->memcpy_d2h_async(out.data(),
                         get_result_buffer(),
                         get_num_elements(),
                         evt);
}

std::size_t
quantity_transformation::get_num_elements() const
{
  return _num_elements;
}


void
quantity_transformation::queue_input_quantities(const std::vector<device_scalar>& input_elements)
{
  assert(input_elements.size() == _input_quantities.size());
  queue_input_quantities(input_elements.data());
}

void
quantity_transformation::queue_input_quantities(const device_scalar* input_elements)
{
  assert(_num_elements + 1 <= _blocksize);
  for(std::size_t i = 0; i < _input_quantities.size(); ++i)
    _input_quantities[i][_num_elements] =
        static_cast<device_scalar>(_scaling_factors[i] * input_elements[i]);

  ++_num_elements;
}

void
quantity_transformation::clear()
{
  _num_elements = 0;
}

void
quantity_transformation::commit_data()
{
  for(std::size_t i = 0; i < _input_quantities.size(); ++i)
    _ctx->memcpy_h2d_async(_input_buffers[i],
                           _input_quantities[i].data(),
                           _num_elements,
                           &_transfers_complete_events[i]);
}

void
quantity_transformation::operator()(cl::Event* evt) const
{
  qcl::kernel_ptr kernel = _quantity.get_kernel(_ctx);

  qcl::kernel_argument_list args{kernel};
  args.push(_result);
  args.push(static_cast<cl_uint>(_num_elements));
  for(std::size_t i = 0; i < _input_buffers.size(); ++i)
    args.push(_input_buffers[i]);
  _quantity.push_additional_kernel_args(args);

  cl_int err =_ctx->get_command_queue().enqueueNDRangeKernel(*kernel,
                                                             cl::NullRange,
                                                             cl::NDRange(math::make_multiple_of(
                                                                           local_size,
                                                                           _num_elements)),
                                                             cl::NDRange(local_size),
                                                             &_transfers_complete_events, evt);
  qcl::check_cl_error(err, "Could not enqueue quantity transformation kernel.");
}

std::size_t
quantity_transformation::get_num_quantities() const
{
  return _input_quantities.size();
}


}
}
