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
#include <boost/program_options.hpp>

#include "cl_types.hpp"
#include "math.hpp"
#include "hdf5_io.hpp"
#include "qcl.hpp"
#include "chandra.hpp"
#include "units.hpp"
#include "gaunt_factor.hpp"


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

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const {}

  virtual bool is_integrated_quantity() const
  {
    return false;
  }

  virtual ~quantity(){}

  virtual const unit_converter& get_unit_converter() const = 0;

};

class illustris_quantity : public quantity
{
public:
  illustris_quantity(const io::illustris_data_loader* data,
                     const std::vector<std::string>& dataset_identifiers,
                     const unit_converter& converter)
    : _data{data},
      _dataset_identifiers{dataset_identifiers},
      _converter{converter}
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

  const unit_converter& get_unit_converter() const override
  {
    return _converter;
  }
private:
  const io::illustris_data_loader* _data;
  std::vector<std::string> _dataset_identifiers;
  unit_converter _converter;
};

class density_temperature_based_quantity : public illustris_quantity
{
public:
  density_temperature_based_quantity(const io::illustris_data_loader* data,
                                     const unit_converter& converter)
      : illustris_quantity{
            data,
            std::vector<std::string>{
              io::illustris_data_loader::get_density_identifier(),
              io::illustris_data_loader::get_internal_energy_identifier()
            },
            converter
          }
  {}

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{
      {
        this->get_unit_converter().density_conversion_factor(),
        1.0
      }
    };
  }

  virtual ~density_temperature_based_quantity(){}
};

class density_temperature_electron_abundance_based_quantity : public illustris_quantity
{
public:
  density_temperature_electron_abundance_based_quantity(const io::illustris_data_loader* data,
                                                        const unit_converter& converter)
      : illustris_quantity{
            data,
            {{io::illustris_data_loader::get_density_identifier(),
              io::illustris_data_loader::get_internal_energy_identifier(),
              io::illustris_data_loader::get_electron_abundance_identifier()}},
            converter
        }
  {
  }

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{
      {
        this->get_unit_converter().density_conversion_factor(),
        1.0, 1.0
      }
    };
  }

  virtual ~density_temperature_electron_abundance_based_quantity(){}
};

class xray_flux_based_quantity : public density_temperature_electron_abundance_based_quantity
{
public:
  xray_flux_based_quantity(const io::illustris_data_loader* data,
                           const unit_converter& converter,
                           const qcl::device_context_ptr& ctx,
                           math::scalar redshift,
                           math::scalar luminosity_distance)
    : density_temperature_electron_abundance_based_quantity{data, converter},
      _z{redshift},
      _luminosity_distance{luminosity_distance},
      _gaunt_factor{ctx}
  {}

  virtual ~xray_flux_based_quantity(){}

  virtual bool is_integrated_quantity() const override
  {
    return true;
  }

  math::scalar get_redshift() const
  {
    return _z;
  }

  math::scalar get_luminosity_distance() const
  {
    return _luminosity_distance;
  }

protected:
  void push_xray_flux_kernel_args(qcl::kernel_argument_list& args) const
  {
    args.push(static_cast<device_scalar>(_z));
    args.push(static_cast<device_scalar>(_luminosity_distance));
    args.push(_gaunt_factor.get_tabulated_function_values());
  }

private:
  math::scalar _z;
  math::scalar _luminosity_distance;

  model::gaunt::thermally_averaged_ff _gaunt_factor;
};

class chandra_xray_total_count_rate : public xray_flux_based_quantity
{
public:
  chandra_xray_total_count_rate(const io::illustris_data_loader* data,
                        const unit_converter& converter,
                        const qcl::device_context_ptr& ctx,
                        math::scalar redshift,
                        math::scalar luminosity_distance)
      : xray_flux_based_quantity{
          data,
          converter,
          ctx,
          redshift,
          luminosity_distance
        },
        _arf{ctx}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("chandra_xray_total_count_rate");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);
    // The kernel also needs the arf
    args.push(_arf.get_tabulated_function_values());
    args.push(_arf.get_min_x());
    args.push(static_cast<cl_int>(_arf.get_num_function_values()));
    args.push(_arf.get_dx());
  }

  virtual ~chandra_xray_total_count_rate(){}

private:
  chandra::arf _arf;
};

class chandra_xray_spectral_count_rate : public xray_flux_based_quantity
{
public:
  chandra_xray_spectral_count_rate(const io::illustris_data_loader* data,
                        const unit_converter& converter,
                        const qcl::device_context_ptr& ctx,
                        math::scalar redshift,
                        math::scalar luminosity_distance,
                        math::scalar photon_energy,
                        math::scalar photon_energy_bin_width)
      : xray_flux_based_quantity{
          data,
          converter,
          ctx,
          redshift,
          luminosity_distance
        },
        _arf{ctx},
        _energy{photon_energy},
        _energy_bin_width{photon_energy_bin_width}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("chandra_xray_spectral_count_rate");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);
    // The kernel also needs the arf
    args.push(_arf.get_tabulated_function_values());
    args.push(_arf.get_min_x());
    args.push(static_cast<cl_int>(_arf.get_num_function_values()));
    args.push(_arf.get_dx());
    args.push(static_cast<math::scalar>(_energy));
    args.push(static_cast<math::scalar>(_energy_bin_width));
  }

  virtual ~chandra_xray_spectral_count_rate(){}

private:
  chandra::arf _arf;
  math::scalar _energy;
  math::scalar _energy_bin_width;
};


class xray_flux : public xray_flux_based_quantity
{
public:
  xray_flux(const io::illustris_data_loader* data,
            const unit_converter& converter,
            const qcl::device_context_ptr& ctx,
            math::scalar redshift,
            math::scalar luminosity_distance,
            math::scalar min_energy,
            math::scalar max_energy,
            unsigned num_samples)
    : xray_flux_based_quantity{
        data,
        converter,
        ctx,
        redshift,
        luminosity_distance
      },
      _min_energy{min_energy},
      _max_energy{max_energy},
      _num_samples{num_samples}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("xray_flux");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);

    args.push(static_cast<device_scalar>(_min_energy));
    args.push(static_cast<device_scalar>(_max_energy));
    args.push(static_cast<cl_int>(_num_samples));
  }

  virtual ~xray_flux(){}

private:
  math::scalar _min_energy;
  math::scalar _max_energy;
  unsigned _num_samples;
};


class xray_spectral_flux : public xray_flux_based_quantity
{
public:
  xray_spectral_flux(const io::illustris_data_loader* data,
                     const unit_converter& converter,
                     const qcl::device_context_ptr& ctx,
                     math::scalar redshift,
                     math::scalar luminosity_distance,
                     math::scalar E,
                     math::scalar dE)
    : xray_flux_based_quantity{
        data,
        converter,
        ctx,
        redshift,
        luminosity_distance
      },
      _photon_energy{E},
      _photon_energy_bin_width{dE}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("xray_spectral_flux");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);

    args.push(static_cast<device_scalar>(_photon_energy));
    args.push(static_cast<device_scalar>(_photon_energy_bin_width));
  }

  virtual ~xray_spectral_flux(){}

private:
  math::scalar _photon_energy;
  math::scalar _photon_energy_bin_width;

};

/*
class luminosity_weighted_temperature : public density_temperature_electron_abundance_based_quantity
{
public:
  luminosity_weighted_temperature(const io::illustris_data_loader* data,
                                  const unit_converter& converter)
      : density_temperature_electron_abundance_based_quantity{data, converter}
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
};*/

class mean_temperature : public density_temperature_electron_abundance_based_quantity
{
public:
  mean_temperature(const io::illustris_data_loader* data, const unit_converter& converter)
      : density_temperature_electron_abundance_based_quantity{data, converter}
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
  interpolation_weight(const io::illustris_data_loader* data,
                       const unit_converter& converter)
      : density_temperature_based_quantity{data, converter}
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
