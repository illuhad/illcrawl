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

#ifndef ILLCRAWL_APP_HPP
#define ILLCRAWL_APP_HPP

#include <memory>
#include <boost/program_options.hpp>

#include "camera.hpp"
#include "quantity.hpp"
#include "particle_distribution.hpp"
#include "environment.hpp"
#include "master_ostream.hpp"
#include "tree_ostream.hpp"
#include "math.hpp"
#include "reconstruction_backend.hpp"
#include "reconstructing_data_crawler.hpp"
#include "fits.hpp"

namespace illcrawl {

class illcrawl_app
{
public:
  illcrawl_app(const illcrawl_app&) = delete;
  illcrawl_app& operator=(const illcrawl_app&) = delete;

  illcrawl_app(int& argc,
               char**& argv,
               std::ostream& output_stream,
               const boost::program_options::options_description& additional_options,
               const math::vector3& periodic_wraparound = {75000.0, 75000.0, 75000.0});

  void parse_command_line();

  const boost::program_options::options_description& get_command_line_options() const;

  math::vector3 get_gas_distribution_center() const;
  math::vector3 get_gas_distribution_size() const;

  const environment& get_environment() const;
  environment& get_environment();


  camera create_distribution_centered_camera(std::size_t resolution_x,
                                             std::size_t resolution_y,
                                             const math::vector3& look_at = {{0., 0., 1.}}) const;

  volume_cutout get_gas_distribution_volume_cutout() const;

  math::scalar get_recommended_integration_depth() const;

  const io::illustris_gas_data_loader& get_data_loader() const;
  const unit_converter& get_unit_converter() const;

  math::scalar get_redshift() const;

  math::scalar get_luminosity_distance() const;

  util::master_ostream& output_stream() const;

  void save_settings_to_fits(util::fits_header* header) const;

  void save_camera_settings_to_fits_header(util::fits_header* header,
                                           const std::string& camera_name,
                                           const camera& camera_configuration) const;

  std::unique_ptr<reconstruction_backend>
  create_voronoi_reconstruction_backend() const;

  std::unique_ptr<reconstruction_backend>
  create_dm_reconstruction_backend() const;

  std::unique_ptr<reconstruction_backend>
  create_reconstruction_backend(const reconstruction_quantity::quantity& q) const;

private:
  void save_length_scalar_to_fits_header(util::fits_header* header,
                                         const std::string& parent_key,
                                         const std::string& key,
                                         math::scalar value,
                                         const std::string& comment) const;

  void save_vector3_to_fits_header(util::fits_header* header,
                                   const std::string& key,
                                   const math::vector3& v,
                                   const std::string& comment) const;

  void save_comoving_vector3_to_fits_header(util::fits_header* header,
                                   const std::string& parent_key,
                                   const std::string& key,
                                   const math::vector3& comoving_vector,
                                   const std::string& comment) const;


  int& _argc;
  char**& _argv;

  environment _env;

  mutable util::master_ostream _master_ostream;

  const math::vector3 _periodic_wraparound;

  std::unique_ptr<particle_distribution> _particle_distribution;
  std::unique_ptr<io::illustris_gas_data_loader> _data_loader;
  std::unique_ptr<unit_converter> _units;

  math::scalar _a = 1/1.2;
  math::scalar _z = 0.2;
  math::scalar _h = 0.704;
  math::scalar _luminosity_distance = 978500.0;

  math::scalar _distribution_radius;

  boost::program_options::options_description _options;

  std::string _hdf5_filename;


  std::string _voronoi_reconstructor = "nn8";
  std::string _dm_reconstructor = "nn8";
  math::scalar _tree_opening_angle = 0.4;

};

class quantity_command_line_parser
{
public:
  quantity_command_line_parser()
  {}

  void register_options(boost::program_options::options_description& options);
  std::unique_ptr<reconstruction_quantity::quantity> create_quantity(const illcrawl_app& app) const;
  void save_configuration_to_fits_header(util::fits_header* header) const;

private:
  void assert_valid_integration_range(math::scalar min,
                                      math::scalar max) const;

  template<class T>
  void assert_greater_equal_zero(T x,  const std::string& message) const
  {
    if(x < static_cast<T>(0))
      throw std::invalid_argument(message);
  }

  template<class T>
  void assert_greater_zero(T x,  const std::string& message) const
  {
    if(x <= static_cast<T>(0))
      throw std::invalid_argument(message);
  }

  template<class T>
  void assert_greater(T a, T b, const std::string& message) const
  {
    if(a <= b)
      throw std::invalid_argument(message);
  }

  std::string _quantity_selection = "chandra_count_rate";

  math::scalar _xray_flux_min_energy = 0.1;
  math::scalar _xray_flux_max_energy = 10.0;
  unsigned _xray_flux_num_samples = 100;

  math::scalar _xray_photon_flux_min_energy = 0.1;
  math::scalar _xray_photon_flux_max_energy = 10.0;
  unsigned _xray_photon_flux_num_samples = 100;

  math::scalar _xray_spectral_flux_energy = 1.0;
  math::scalar _xray_spectral_flux_energy_bin_width = 0.1;

  math::scalar _chandra_spectral_count_rate_energy = 1.0;
  math::scalar _chandra_spectral_count_rate_energy_bin_width = 0.1;

  math::scalar _luminosity_weighted_temp_min_energy = 0.1;
  math::scalar _luminosity_weighted_temp_max_energy = 10.0;
};

}

#endif
