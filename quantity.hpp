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


#ifndef QUANTITY_HPP
#define QUANTITY_HPP

#include <cassert>
#include <vector>

#include "math.hpp"
#include "hdf5_io.hpp"
#include "qcl.hpp"

namespace illcrawl {
namespace reconstruction_quantity {

class quantity
{
public:

  virtual std::vector<H5::DataSet> get_required_datasets() const = 0;
  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const
  {
    return std::vector<math::scalar>(get_required_datasets().size(), 1.0);
  }
  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const = 0;

  virtual bool is_integrated_quantity() const
  {
    return false;
  }

  virtual ~quantity(){}
};

class illustris_quantity : public quantity
{
public:
  illustris_quantity(const io::illustris_data_loader* data,
                     const std::vector<std::string>& dataset_identifiers)
    : _data{data}, _dataset_identifiers{dataset_identifiers}
  {
    assert(_data != nullptr);
  }

  virtual std::vector<H5::DataSet> get_required_datasets() const override
  {
    std::vector<H5::DataSet> result;
    for(std::string dataset_name : _dataset_identifiers)
      result.push_back(_data->get_dataset(dataset_name));
    return result;
  }

  virtual ~illustris_quantity() {}

private:
  const io::illustris_data_loader* _data;
  std::vector<std::string> _dataset_identifiers;
};

class density_temperature_based_quantity : public illustris_quantity
{
public:
  density_temperature_based_quantity(const io::illustris_data_loader* data)
      : illustris_quantity{
            data,
            {{io::illustris_data_loader::get_density_identifier(),
              io::illustris_data_loader::get_internal_energy_identifier()}}}
  {
  }

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{{1.e7, 1.0}};
  }

  virtual ~density_temperature_based_quantity(){}
};

class density_temperature_electron_abundance_based_quantity : public illustris_quantity
{
public:
  density_temperature_electron_abundance_based_quantity(const io::illustris_data_loader* data)
      : illustris_quantity{
            data,
            {{io::illustris_data_loader::get_density_identifier(),
              io::illustris_data_loader::get_internal_energy_identifier(),
              io::illustris_data_loader::get_electron_abundance_identifier()}}}
  {
  }

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{{1.e7, 1.0, 1.0}};
  }

  virtual ~density_temperature_electron_abundance_based_quantity(){}
};

class xray_emission : public density_temperature_electron_abundance_based_quantity
{
public:
  xray_emission(const io::illustris_data_loader* data)
      : density_temperature_electron_abundance_based_quantity{data}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("xray_emission");
  }

  virtual bool is_integrated_quantity() const override
  {
    return true;
  }

  virtual ~xray_emission(){}
};


class luminosity_weighted_temperature : public density_temperature_electron_abundance_based_quantity
{
public:
  luminosity_weighted_temperature(const io::illustris_data_loader* data)
      : density_temperature_electron_abundance_based_quantity{data}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("luminosity_weighted_temperature");
  }

  virtual bool is_integrated_quantity() const override
  {
    return true;
  }

  virtual ~luminosity_weighted_temperature(){}
};

class mean_temperature : public density_temperature_electron_abundance_based_quantity
{
public:
  mean_temperature(const io::illustris_data_loader* data)
      : density_temperature_electron_abundance_based_quantity{data}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("mean_temperature");
  }

  virtual ~mean_temperature(){}
};

class interpolation_weight : public density_temperature_based_quantity
{
public:
  interpolation_weight(const io::illustris_data_loader* data)
      : density_temperature_based_quantity{data}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("identity");
  }

  virtual ~interpolation_weight(){}
};


class quantity_transformation
{
public:

  quantity_transformation(const qcl::device_context_ptr& ctx,
                          const quantity& q,
                          std::size_t blocksize)
    : _ctx{ctx}, _quantity{q}, _blocksize{blocksize}, _num_elements{0}
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

  quantity_transformation(const quantity_transformation&) = delete;
  quantity_transformation& operator=(const quantity_transformation&) = delete;

  const cl::Buffer& get_result_buffer() const
  {
    return _result;
  }

  void retrieve_results(cl::Event* evt,
                        std::vector<device_scalar>& out) const
  {
    out.resize(get_num_elements());
    _ctx->memcpy_d2h_async(out.data(),
                           get_result_buffer(),
                           get_num_elements(),
                           evt);
  }

  std::size_t get_num_elements() const
  {
    return _num_elements;
  }

  template<std::size_t N>
  void queue_input_quantities(const std::array<device_scalar, N>& input_elements)
  {
    assert(N == _input_quantities.size());
    queue_input_quantities(input_elements.data());
  }

  void queue_input_quantities(const std::vector<device_scalar>& input_elements)
  {
    assert(input_elements.size() == _input_quantities.size());
    queue_input_quantities(input_elements.data());
  }

  void queue_input_quantities(const device_scalar* input_elements)
  {
    assert(_num_elements + 1 <= _blocksize);
    for(std::size_t i = 0; i < _input_quantities.size(); ++i)
      _input_quantities[i][_num_elements] =
          static_cast<device_scalar>(_scaling_factors[i] * input_elements[i]);

    ++_num_elements;
  }

  void clear()
  {
    _num_elements = 0;
  }

  void commit_data()
  {
    for(std::size_t i = 0; i < _input_quantities.size(); ++i)
      _ctx->memcpy_h2d_async(_input_buffers[i],
                             _input_quantities[i].data(),
                             _num_elements,
                             &_transfers_complete_events[i]);
  }

  void operator()(cl::Event* evt) const
  {
    qcl::kernel_ptr kernel = _quantity.get_kernel(_ctx);

    qcl::kernel_argument_list args{kernel};
    args.push(_result);
    args.push(static_cast<cl_uint>(_num_elements));
    for(std::size_t i = 0; i < _input_buffers.size(); ++i)
      args.push(_input_buffers[i]);

    cl_int err =_ctx->get_command_queue().enqueueNDRangeKernel(*kernel,
                                                               cl::NullRange,
                                                               cl::NDRange(math::make_multiple_of(
                                                                 local_size,
                                                                 _num_elements)),
                                                               cl::NDRange(local_size),
                                                               &_transfers_complete_events, evt);
    qcl::check_cl_error(err, "Could not enqueue quantity transformation kernel.");
  }

  std::size_t get_num_quantities() const
  {
    return _input_quantities.size();
  }
private:
  static constexpr std::size_t local_size = 256;

  qcl::device_context_ptr _ctx;
  const quantity& _quantity;
  std::size_t _blocksize;

  std::vector<std::vector<device_scalar>> _input_quantities;
  std::vector<cl::Buffer> _input_buffers;

  std::size_t _num_elements;

  cl::Buffer _result;

  std::vector<cl::Event> _transfers_complete_events;
  std::vector<math::scalar> _scaling_factors;
};

}
}

#endif
