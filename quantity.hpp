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

}
}

#endif
