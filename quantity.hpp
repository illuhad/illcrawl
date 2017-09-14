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

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const = 0;
  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const = 0;


  virtual ~quantity(){}

  virtual const unit_converter& get_unit_converter() const = 0;

  virtual bool is_quantity_baryonic() const = 0;

};

class illustris_quantity : public quantity
{
public:
  illustris_quantity(io::illustris_data_loader* data,
                     const std::vector<std::string>& dataset_identifiers,
                     const unit_converter& converter,
                     std::size_t particle_type_id = 0)
    : _data{data},
      _dataset_identifiers{dataset_identifiers},
      _converter{converter},
      _particle_type_id{particle_type_id}
  {
    _data->select_group(_particle_type_id);
    assert(_data != nullptr);
  }

  virtual std::vector<H5::DataSet> get_required_datasets() const override
  {
    if(_data->get_current_group_name() != ("PartType"+std::to_string(_particle_type_id)))
      _data->select_group(_particle_type_id);

    std::vector<H5::DataSet> result;
    for(std::string dataset_name : _dataset_identifiers)
      result.push_back(_data->get_dataset(dataset_name));
    return result;
  }

  virtual ~illustris_quantity() {}

  virtual const unit_converter& get_unit_converter() const override
  {
    return _converter;
  }

  virtual bool is_quantity_baryonic() const override
  {
    return true;
  }
private:
  io::illustris_data_loader* _data;
  std::vector<std::string> _dataset_identifiers;
  unit_converter _converter;
  std::size_t _particle_type_id;
};

class density_based_quantity : public illustris_quantity
{
public:
  density_based_quantity(io::illustris_data_loader* data,
                         const unit_converter& converter)
      : illustris_quantity{
            data,
            std::vector<std::string>{
              io::illustris_data_loader::get_density_identifier()
            },
            converter
          }
  {}

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{
      {
        this->get_unit_converter().density_conversion_factor()
      }
    };
  }

  virtual ~density_based_quantity(){}
};

class density_temperature_electron_abundance_based_quantity : public illustris_quantity
{
public:
  density_temperature_electron_abundance_based_quantity(io::illustris_data_loader* data,
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
  xray_flux_based_quantity(io::illustris_data_loader* data,
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

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    return dV;
  }
  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    return dA;
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
  chandra_xray_total_count_rate(io::illustris_data_loader* data,
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
  chandra_xray_spectral_count_rate(io::illustris_data_loader* data,
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


enum class flux_type
{
  XRAY_FLUX,
  XRAY_PHOTON_FLUX
};

template<flux_type Flux_type>
class flux : public xray_flux_based_quantity
{
public:
  flux(io::illustris_data_loader* data,
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
    if(Flux_type == flux_type::XRAY_FLUX)
      return ctx->get_kernel("xray_flux");
    else if(Flux_type == flux_type::XRAY_PHOTON_FLUX)
      return ctx->get_kernel("xray_photon_flux");
    else
      throw std::runtime_error("Invalid flux type!");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);

    args.push(static_cast<device_scalar>(_min_energy));
    args.push(static_cast<device_scalar>(_max_energy));
    args.push(static_cast<cl_int>(_num_samples));
  }

  virtual ~flux(){}

private:
  math::scalar _min_energy;
  math::scalar _max_energy;
  unsigned _num_samples;
};

using xray_flux        = flux<flux_type::XRAY_FLUX>;
using xray_photon_flux = flux<flux_type::XRAY_PHOTON_FLUX>;


class xray_spectral_flux : public xray_flux_based_quantity
{
public:
  xray_spectral_flux(io::illustris_data_loader* data,
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


class luminosity_weighted_temperature : public xray_flux_based_quantity
{
public:
  luminosity_weighted_temperature(io::illustris_data_loader* data,
                                  const unit_converter& converter,
                                  const qcl::device_context_ptr& ctx,
                                  math::scalar redshift,
                                  math::scalar luminosity_distance,
                                  math::scalar min_energy,
                                  math::scalar max_energy)
      : xray_flux_based_quantity{
          data,
          converter,
          ctx,
          redshift,
          luminosity_distance
        },
        _min_energy{min_energy},
        _max_energy{max_energy}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("luminosity_weighted_temperature");
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    this->push_xray_flux_kernel_args(args);

    args.push(static_cast<device_scalar>(_min_energy));
    args.push(static_cast<device_scalar>(_max_energy));
    args.push(static_cast<cl_int>(100));
  }

  virtual ~luminosity_weighted_temperature(){}

private:
  math::scalar _min_energy;
  math::scalar _max_energy;
};

class mean_temperature : public density_temperature_electron_abundance_based_quantity
{
public:
  mean_temperature(io::illustris_data_loader* data,
                   const unit_converter& converter)
      : density_temperature_electron_abundance_based_quantity{data, converter}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("mean_temperature");
  }

  virtual ~mean_temperature(){}

  virtual math::scalar effective_volume_integration_dV(math::scalar dV, math::scalar integration_volume) const override
  {
    return dV / integration_volume;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA, math::scalar integration_range) const override
  {
    // We want to obtain the mean temperature along the line of sight, therefore
    // we need to calculate T = Integral dz*T(z) / (Integral dz) => dA_eff = 1/integration_range
    return 1.0 / integration_range;
  }
};

class interpolation_weight : public density_based_quantity
{
public:
  interpolation_weight(io::illustris_data_loader* data,
                       const unit_converter& converter)
      : density_based_quantity{data, converter}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr& ctx) const override
  {
    return ctx->get_kernel("identity");
  }

  virtual ~interpolation_weight(){}

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    return dV / integration_volume;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    return 1.0 / integration_range;
  }
};

class mean_density : public density_based_quantity
{
public:
  mean_density(io::illustris_data_loader* data,
               const unit_converter& converter)
    : density_based_quantity{data, converter}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr &ctx) const override
  {
    return ctx->get_kernel("unprocessed_quantity");
  }

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    return dV / integration_volume;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    return 1.0 / integration_range;
  }

  virtual ~mean_density(){}
};

class mass : public density_based_quantity
{
public:
  mass(io::illustris_data_loader* data,
       const unit_converter& converter)
      : density_based_quantity{data, converter}
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr &ctx) const override
  {
    return ctx->get_kernel("unprocessed_quantity");
  }

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    return dV;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    return dA;
  }

  virtual ~mass(){}
};


class potential : public illustris_quantity
{
public:
  potential(io::illustris_data_loader* data,
            const unit_converter& converter)
      : illustris_quantity{
          data,
          std::vector<std::string>{
            "Potential"
          },
          converter
        }
  {}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr &ctx) const override
  {
    return ctx->get_kernel("unprocessed_quantity");
  }

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{{
        this->get_unit_converter().potential_conversion_factor()
    }};
  }

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    // mean potential
    return dV / integration_volume;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    // integrated potential along line of sight. We do not need dA because we
    // are not integrating over the area, and we do not need the integration_range
    // because we are not calculating the mean, but the projected (integrated)
    // potential.
    return 1.0;
  }

  virtual ~potential(){}
};

class dm_quantity : public illustris_quantity
{
public:
  dm_quantity(io::illustris_data_loader* data,
             const unit_converter& converter,
             math::scalar dm_particle_mass ) // DM particle mass in 10^10 Msun/h
    : illustris_quantity{data, std::vector<std::string>{}, converter, 1},
      _dm_particle_mass{converter.mass_conversion_factor() * dm_particle_mass}
  {}

  virtual bool is_quantity_baryonic() const override
  {
    return false;
  }

  virtual ~dm_quantity(){}

  virtual qcl::kernel_ptr get_kernel(const qcl::device_context_ptr &ctx) const override
  {
    return ctx->get_kernel("constant_quantity");
  }

  virtual std::vector<math::scalar> get_quantitiy_scaling_factors() const override
  {
    return std::vector<math::scalar>{};
  }

  virtual void push_additional_kernel_args(qcl::kernel_argument_list& args) const override
  {
    args.push(static_cast<device_scalar>(_dm_particle_mass));
  }

  math::scalar get_dm_particle_mass() const
  {
    return _dm_particle_mass;
  }


private:
  math::scalar _dm_particle_mass;

};


class dm_density : public dm_quantity
{
public:
  dm_density(io::illustris_data_loader* data,
             const unit_converter& converter,
             math::scalar dm_particle_mass) // DM particle mass in 10^10 Msun/h
    : dm_quantity{data, converter, dm_particle_mass}
  {}

  virtual ~dm_density(){}

  virtual math::scalar effective_volume_integration_dV(math::scalar dV,
                                                       math::scalar integration_volume) const override
  {
    // mean density
    return dV / integration_volume;
  }

  virtual math::scalar effective_line_of_sight_integration_dA(math::scalar dA,
                                                              math::scalar integration_range) const override
  {
    // mean density
    return 1.0/integration_range;
  }


};



class quantity_transformation
{
public:

  quantity_transformation(const qcl::device_context_ptr& ctx,
                          const quantity& q,
                          std::size_t blocksize);

  quantity_transformation(const quantity_transformation&) = delete;
  quantity_transformation& operator=(const quantity_transformation&) = delete;

  const cl::Buffer& get_result_buffer() const;

  void retrieve_results(cl::Event* evt,
                        std::vector<device_scalar>& out) const;

  std::size_t get_num_elements() const;

  template<std::size_t N>
  void queue_input_quantities(const std::array<device_scalar, N>& input_elements)
  {
    assert(N == _input_quantities.size());
    queue_input_quantities(input_elements.data());
  }

  void queue_input_quantities(const std::vector<device_scalar>& input_elements);

  void queue_input_quantities(const device_scalar* input_elements);

  void clear();

  void commit_data();

  void operator()(cl::Event* evt) const;

  std::size_t get_num_quantities() const;
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
