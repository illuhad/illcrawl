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

#include <memory>
#include <boost/program_options.hpp>

#include "illcrawl_app.hpp"
#include "volumetric_reconstruction_backend_nn8.hpp"
#include "volumetric_reconstruction_backend_tree.hpp"
#include "dm_reconstruction_backend_brute_force.hpp"
#include "dm_reconstruction_backend_grid.hpp"
#include "projective_smoothing_reconstruction.hpp"
#include "projective_smoothing_backend_grid.hpp"

namespace illcrawl {

/************ Implementation of illcrawl_app ***************/

illcrawl_app::illcrawl_app(int& argc,
                           char**& argv,
                           std::ostream& output_stream,
                           const boost::program_options::options_description& additional_options,
                           const math::vector3& periodic_wraparound)
  : _argc{argc}, _argv{argv},
    _env{argc, argv},
    _master_ostream{
      output_stream,
      _env.get_communicator(),
      _env.get_master_rank()
    },
    _periodic_wraparound(periodic_wraparound)
{

  _options.add_options()
      ("input,i", boost::program_options::value<std::string>(&_hdf5_filename)->required(),
       "the hdf5 file containing the data")
      ("hubble_constant,h", boost::program_options::value<math::scalar>(&_h)->default_value(_h), "Hubble constant in units of 100 km/s/Mpc")
      ("redshift,z", boost::program_options::value<math::scalar>(&_z)->default_value(_z), "Redshift")
      ("luminosity_distance,d", boost::program_options::value<math::scalar>(&_luminosity_distance)->default_value(_luminosity_distance),
       "Luminosity distance of the dataset in kpc (e.g. z=0.2 in Illustris-1 cosmology: dL=978500 kpc)")
      ("reconstructor.voronoi", boost::program_options::value<std::string>(&_voronoi_reconstructor)->default_value(_voronoi_reconstructor),
       "The reconstructor used for quantities that rely on voronoi meshes (baryons). Supported reconstructors:\n"
       " * nn8\n"
       " * tree")
      ("reconstructor.voronoi.tree.opening_angle",
       boost::program_options::value<math::scalar>(&_tree_opening_angle)->default_value(_tree_opening_angle),
       "The opening angle of the tree reconstructor.")
      ("reconstructor.smoothing",
       boost::program_options::value<std::string>(&_smoothing_reconstructor)->default_value(_smoothing_reconstructor),
       "The reconstructor used for quantities that rely on particle smoothing (e.g. Dark Matter). Supported reconstructors:\n"
       " * brute_force\n"
       " * grid")
      ("reconstructor.smoothing.projective",
       boost::program_options::value<std::string>(&_projective_smoothing_reconstructor)->default_value(_projective_smoothing_reconstructor),
       "The projective reconstructor used to project smoothed quantities. Supported reconstructors:\n"
       " * grid")
      ("data_crawler.blocksize",
       boost::program_options::value<std::size_t>(&_data_crawling_blocksize)->default_value(_data_crawling_blocksize),
       "The number of particles (or cells) that are read and processed in one block.");


  _options.add(additional_options);


  _master_ostream << "Detected system configuration:" << std::endl;

  illcrawl::util::tree_formatter tree_output{"Root"};
  for(const auto& node : _env.get_node_rank_map())
  {
    std::string node_entry_name = node.first;
    illcrawl::util::tree_formatter::node node_entry;
    for(int rank = 0; rank < static_cast<int>(node.second.size()); ++rank)
    {
      int global_rank = node.second[rank];
      std::string process_entry_name = "Local rank "
          +std::to_string(rank)
          +" (Global rank "+std::to_string(global_rank)
          +")";

      illcrawl::util::tree_formatter::node process_entry;
      process_entry.append_content("Using device: " + _env.get_device_names()[global_rank]+"\n");
      process_entry.append_content("Device capabilities: " + _env.get_device_extensions()[global_rank]+"\n");

      node_entry.add_node(process_entry_name, process_entry);
    }
    tree_output.get_root().add_node(node_entry_name, node_entry);
  }
  _master_ostream << tree_output << std::endl;
}

void
illcrawl_app::parse_command_line()
{

  boost::program_options::variables_map vm;
  boost::program_options::store(
        boost::program_options::command_line_parser(_argc, _argv).options(_options)
        //.allow_unregistered()
        .run(),
        vm);

  boost::program_options::notify(vm);

  _a = 1.0 / (1. + _z);
  _units = std::unique_ptr<unit_converter>{new unit_converter{_a, _h}};
  _data_loader = std::unique_ptr<io::illustris_data_loader>{
      new io::illustris_data_loader{_hdf5_filename, 0}
  };

  _particle_distribution = std::unique_ptr<particle_distribution>{
      new particle_distribution{
        _data_loader->get_dataset(io::illustris_data_loader::get_coordinate_identifier()),
        _periodic_wraparound
      }
  };

  _distribution_radius =
      0.5 * std::sqrt(math::dot(get_gas_distribution_size(), get_gas_distribution_size()));

  _master_ostream << "Gas distribution center: " << get_gas_distribution_center()[0] << ", "
                  << get_gas_distribution_center()[1] << ", "
                  << get_gas_distribution_center()[2]
                  << std::endl;
  _master_ostream << "Gas distribution size: "
                  << get_gas_distribution_size()[0] << "x"
                  << get_gas_distribution_size()[1] << "x"
                  << get_gas_distribution_size()[2] << " (ckpc/h)^3" << std::endl;

}

const boost::program_options::options_description&
illcrawl_app::get_command_line_options() const
{
  return _options;
}

math::vector3
illcrawl_app::get_gas_distribution_center() const
{
  return _particle_distribution->get_extent_center();
}

math::vector3
illcrawl_app::get_gas_distribution_size() const
{
  return _particle_distribution->get_distribution_size();
}

const environment&
illcrawl_app::get_environment() const
{
  return _env;
}

environment&
illcrawl_app::get_environment()
{
  return _env;
}


camera
illcrawl_app::create_distribution_centered_camera(std::size_t resolution_x,
                                                  std::size_t resolution_y,
                                                  const math::vector3& look_at) const
{

  math::vector3 cam_position = get_gas_distribution_center();
  cam_position -= _distribution_radius * look_at;
  return camera{cam_position, look_at, 0.0, get_gas_distribution_size()[0], resolution_x, resolution_y};
}

volume_cutout
illcrawl_app::get_gas_distribution_volume_cutout() const
{
  return volume_cutout{
    get_gas_distribution_center(),
        get_gas_distribution_size(),
        _periodic_wraparound
  };
}

math::scalar
illcrawl_app::get_recommended_integration_depth() const
{
  return 2.0 * _distribution_radius;
}

io::illustris_data_loader&
illcrawl_app::get_data_loader() const
{
  return *_data_loader.get();
}

const unit_converter&
illcrawl_app::get_unit_converter() const
{
  return *_units;
}

math::scalar
illcrawl_app::get_redshift() const
{
  return _z;
}

math::scalar
illcrawl_app::get_luminosity_distance() const
{
  return _luminosity_distance;
}

util::master_ostream&
illcrawl_app::output_stream() const
{
  return _master_ostream;
}

void
illcrawl_app::save_settings_to_fits(util::fits_header* header) const
{

  header->add_entry("renderer","Illcrawl suite");
  header->add_entry("renderer.voronoi_reconstructor",
                    this->_voronoi_reconstructor);
  header->add_entry("renderer.dm_reconstructor",
                    this->_smoothing_reconstructor);
  header->add_entry("renderer.dm_projective_reconstructor",
                    this->_projective_smoothing_reconstructor);
  header->add_entry("renderer.partitioner","uniform","The parallel work partitioning strategy");
  header->add_entry("data_crawler.blocksize",
                    static_cast<unsigned long long>(_data_crawling_blocksize),
                    "The number of particles/cells processed in one block");
  header->add_entry("environment.num_mpi_processes",
                    _env.get_communicator().size(),
                    "Number of parallel MPI processes");
  header->add_entry("environment.device_name",
                    _env.get_compute_device()->get_device_name(),
                    "Name of Compute device of master process");
  header->add_entry("environment.device_vendor",
                    _env.get_compute_device()->get_device_vendor(),
                    "Name of compute device vendor on the master process");
  header->add_entry("environment.device_opencl_version",
                    _env.get_compute_device()->get_device_cl_version(),
                    "The OpenCL version supported by the compute device of the master process");
  header->add_entry("environment.device_driver_version",
                    _env.get_compute_device()->get_driver_version(),
                    "Version of the driver of the compute device of the master process");
  header->add_entry("data_source",_hdf5_filename);
  header->add_entry("a", this->_a, "Scale factor");
  header->add_entry("z", this->_z, "Redshift");
  header->add_entry("h", this->_h, "Hubble constant [100 km/s/Mpc]");
  header->add_entry("dL", this->_luminosity_distance, "Luminosity distance [kpc]");

  save_length_scalar_to_fits_header(header,
                                    "particle_distribution",
                                    "radius",
                                    _distribution_radius,
                                    "Radius within which all particles lie");


  save_comoving_vector3_to_fits_header(header,
                                       "particle_distribution",
                                       "center",
                                       _particle_distribution->get_extent_center(),
                                       "Particle distribution's box center");
  save_comoving_vector3_to_fits_header(header,
                                       "particle_distribution",
                                       "size",
                                       _particle_distribution->get_distribution_size(),
                                       "The size of the distribution");


}

void
illcrawl_app::save_camera_settings_to_fits_header(util::fits_header* header,
                                                  const std::string& camera_name,
                                                  const camera& camera_configuration) const
{
  save_comoving_vector3_to_fits_header(header,
                                       camera_name,
                                       "position",
                                       camera_configuration.get_position(),
                                       "Center position of the camera plane");
  save_comoving_vector3_to_fits_header(header,
                                       camera_name,
                                       "screen_min",
                                       camera_configuration.get_screen_min_position(),
                                       "Lower left corner position of the screen");

  save_vector3_to_fits_header(header,
                              camera_name+".look_at",
                              camera_configuration.get_look_at(),
                              "Camera look-at vector");
  save_vector3_to_fits_header(header,
                              camera_name+".screen_basis0",
                              camera_configuration.get_screen_basis_vector0(),
                              "The basis vector of the x-direction of the camera screen");
  save_vector3_to_fits_header(header,
                              camera_name+".screen_basis1",
                              camera_configuration.get_screen_basis_vector1(),
                              "The basis vector of the y-direction of the camera screen");
  save_length_scalar_to_fits_header(header,
                                    camera_name,
                                    "pixel_size",
                                    camera_configuration.get_pixel_size(),
                                    "The physical width of one pixel");
}

void
illcrawl_app::save_length_scalar_to_fits_header(util::fits_header* header,
                                                const std::string& parent_key,
                                                const std::string& key,
                                                math::scalar value,
                                                const std::string& comment) const
{
  std::string comoving_full_key = parent_key + "." + key;
  std::string proper_full_key = parent_key + ".proper_" + key;

  header->add_entry(comoving_full_key, value, comment + " (Comoving length [ckpc/h])");
  header->add_entry(proper_full_key,
                    _units->length_conversion_factor() * value,
                    comment + " (Proper length [kpc])");
}

void
illcrawl_app::save_vector3_to_fits_header(util::fits_header* header,
                                          const std::string& key,
                                          const math::vector3& v,
                                          const std::string& comment) const
{
  std::array<std::string,3> axes{{"x","y","z"}};

  for(std::size_t i = 0; i < 3; ++i)
  {
    header->add_entry(key+"."+axes[i], v[i],
                      comment);
  }
}

void
illcrawl_app::save_comoving_vector3_to_fits_header(util::fits_header* header,
                                                   const std::string& parent_key,
                                                   const std::string& key,
                                                   const math::vector3& comoving_vector,
                                                   const std::string& comment) const
{
  std::string comoving_full_key = parent_key + "." + key;
  std::string proper_full_key = parent_key + ".proper_" + key;

  save_vector3_to_fits_header(header,
                              comoving_full_key,
                              comoving_vector,
                              comment + " (Comoving length [ckpc/h])");
  save_vector3_to_fits_header(header,
                              proper_full_key,
                              _units->length_conversion_factor() * comoving_vector,
                              comment + " (Proper length [kpc])");
}

std::unique_ptr<reconstruction_backend>
illcrawl_app::create_voronoi_reconstruction_backend(
    const reconstruction_quantity::quantity& q) const
{
  if(this->_voronoi_reconstructor == "nn8")
  {
    return std::unique_ptr<reconstruction_backend>{
      new reconstruction_backends::nn8{_env.get_compute_device()}
    };
  }
  else if(this->_voronoi_reconstructor == "tree")
  {
    return std::unique_ptr<reconstruction_backend>{
      new reconstruction_backends::tree{
        _env.get_compute_device(),
        _tree_opening_angle
      }
    };
  }
  else
    throw std::invalid_argument{"Unknown voronoi reconstructor: "+_voronoi_reconstructor};
}

std::unique_ptr<reconstruction_backend>
illcrawl_app::create_smoothing_reconstruction_backend(
    const reconstruction_quantity::quantity& q) const
{
  assert(!q.requires_voronoi_reconstruction());

  this->open_datagroup_for_quantity(q);

  std::string dm_smoothing_length_id =
      io::illustris_data_loader::get_dm_smoothing_length_identifier();

  if(this->_smoothing_reconstructor == "brute_force")
  {
    return std::unique_ptr<reconstruction_backend>{
      new reconstruction_backends::dm::brute_force{
        _env.get_compute_device(),
        _data_loader->get_dataset(dm_smoothing_length_id)
      }
    };
  }
  else if(this->_smoothing_reconstructor == "grid")
  {
    return std::unique_ptr<reconstruction_backend>{
      new reconstruction_backends::dm::grid{
        _env.get_compute_device(),
        _data_loader->get_dataset(dm_smoothing_length_id)
      }
    };
  }
  else
    throw std::invalid_argument{"Unknown dark matter reconstructor: "+_smoothing_reconstructor};
}


std::unique_ptr<reconstruction_backend>
illcrawl_app::create_reconstruction_backend(const reconstruction_quantity::quantity& q) const
{
  if(q.requires_voronoi_reconstruction())
    return create_voronoi_reconstruction_backend(q);
  else
    return create_smoothing_reconstruction_backend(q);
}


std::unique_ptr<reconstruction_backend>
illcrawl_app::create_projective_smoothing_reconstruction_backend(
    const camera& cam,
    math::scalar max_integration_depth,
    const reconstruction_quantity::quantity* q) const
{
  std::unique_ptr<projective_smoothing_backend> backend;

  this->open_datagroup_for_quantity(*q);

  std::string smoothing_length_dataset_id =
      io::illustris_data_loader::get_dm_smoothing_length_identifier();

  if(this->_projective_smoothing_reconstructor == "grid")
  {
    backend.reset(new reconstruction_backends::dm::projective_smoothing_grid{
                    _env.get_compute_device(),
                    _data_loader->get_dataset(smoothing_length_dataset_id)
                  });
  }
  else
    throw std::invalid_argument{"Unknown projective smoothing reconstruction backend: "+
                                this->_projective_smoothing_reconstructor};


  return std::unique_ptr<reconstruction_backend>{
    new reconstruction_backends::dm::projective_smoothing{
      this->_env.get_compute_device(),
      cam,
      max_integration_depth,
      q,
      std::move(backend)
    }
  };
}

void
illcrawl_app::open_datagroup_for_quantity(const reconstruction_quantity::quantity& q) const
{
  using illustris_quantity_ptr = const reconstruction_quantity::illustris_quantity*;
  illustris_quantity_ptr casted_quantity = nullptr;

  if((casted_quantity = dynamic_cast<illustris_quantity_ptr>(&q)) != nullptr)
    if(_data_loader->get_current_group_name() !=
        ("PartType"+std::to_string(casted_quantity->get_particle_type_id())))
    _data_loader->select_group(casted_quantity->get_particle_type_id());
}

std::unique_ptr<reconstruction_backend>
illcrawl_app::create_projective_smoothing_reconstruction_backend(const reconstruction_quantity::quantity* q) const
{
  return this->create_projective_smoothing_reconstruction_backend(camera{},
                                                           1.0,
                                                           q);
}


std::size_t
illcrawl_app::get_data_crawling_blocksize() const
{
  return this->_data_crawling_blocksize;
}

/*********** Implementation of quantity_command_line_parser ************/


void
quantity_command_line_parser::register_options(boost::program_options::options_description& options)
{
  options.add_options()
      ("quantity,q",
       boost::program_options::value<std::string>(&_quantity_selection)->default_value(_quantity_selection),
       "The quantity which shall be used for reconstruction. Allowed values: \n"
       " * chandra_count_rate: count rate as seen by chandra, taking into account chandra's ACIS-I instrumental response [counts/s]\n"
       " * xray_flux: The emitted xray flux [keV/s/m^2]\n"
       " * xray_photon_flux: The emitted photon flux. Note that unlike xray_flux, this is per cm^2. [photons/s/cm^2]\n"
       " * chandra_spectral_count_rate: The count rate as seen by chandra's ACIS-I within an energy channel [counts/]\n"
       " * xray_spectral_flux: The emitted xray flux within an energy channel [keV/s/m^2]\n"
       " * mean_temperature: The mean temperature along the line of sight [K]\n"
       " * luminosity_weighted_temperature: Calculates xray_flux*temperature along the line of sight. [keV/s/m^2*K]\n"
       " * mean_density: The mean gas density along the line of sight [M_sun/kpc^3]\n"
       " * projected_density: The projected gas density along the line of sight\n"
       "   [M_sun/kpc^2] for projections, [M_sun/kpc^3] for volume integrations.\n"
       " * mass: The total baryonic mass along the line of sight [M_sun]\n"
       " * potential: The gravitational potential [(km/s)^2]\n"
       " * metallicity: M_Z/M_total, i.e. the metal mass divided by the total mass [solar metallicities]\n"
       " * density_weighted_metallicity: The metallicity weighted with the density\n"
       "   [solar metallicites * M_sun/kpc^2] for projections, [solar metallicities*M_sun/kpc^3] otherwise\n"
       " * luminosity_weighted_metallicity: Calculate xray_flux*metallicity along the line of sight [keV/s/m^2*solar metallicities]\n"
       " * dm_mean_density: The dark matter mean density [M_sun/kpc^3]\n"
       " * dm_mass: The dark matter mass [M_sun]\n"
       " * stellar_mean_density: The stellar mean density [M_sun/kpc^3]\n"
       " * stellar_mass: The stellar mass [M_sun]")
      ("quantity.xray_spectral_flux.energy",
       boost::program_options::value<math::scalar>(&_xray_spectral_flux_energy)->default_value(
         _xray_spectral_flux_energy),
       "The energy for the xray_spectral_flux quantity [keV]")
      ("quantity.xray_spectral_flux.energy_bin_width",
       boost::program_options::value<math::scalar>(&_xray_spectral_flux_energy_bin_width)->default_value(
         _xray_spectral_flux_energy_bin_width),
       "The energy bin width of the xray_spectral_flux quantity [keV]")
      ("quantity.chandra_spectral_count_rate.energy",
       boost::program_options::value<math::scalar>(&_chandra_spectral_count_rate_energy)->default_value(
         _chandra_spectral_count_rate_energy),
       "The energy for the chandra_spectral_count_rate quantity [keV]")
      ("quantity.chandra_spectral_count_rate.energy_bin_width",
       boost::program_options::value<math::scalar>(&_chandra_spectral_count_rate_energy_bin_width)->default_value(
         _chandra_spectral_count_rate_energy_bin_width),
       "The energy bin width of the chandra_spectral_count_rate quantity [keV]")
      ("quantity.xray_flux.min_energy",
       boost::program_options::value<math::scalar>(&_xray_flux_min_energy)->default_value(
         _xray_flux_min_energy),
       "Start of energy integration range for the xray_flux quantity [keV]")
      ("quantity.xray_flux.max_energy",
       boost::program_options::value<math::scalar>(&_xray_flux_max_energy)->default_value(
         _xray_flux_max_energy),
       "End of energy integration range for the xray_flux quantity [keV]")
      ("quantity.xray_flux.num_samples",
       boost::program_options::value<unsigned>(&_xray_flux_num_samples)->default_value(
         _xray_flux_num_samples),
       "Number of samples for the energy integration of the xray_flux quantity")
      ("quantity.xray_photon_flux.min_energy",
       boost::program_options::value<math::scalar>(&_xray_photon_flux_min_energy)->default_value(
         _xray_photon_flux_min_energy),
       "Start of energy integration range for the xray_photon_flux quantity [keV]")
      ("quantity.xray_photon_flux.max_energy",
       boost::program_options::value<math::scalar>(&_xray_photon_flux_max_energy)->default_value(
         _xray_photon_flux_max_energy),
       "End of energy integration range for the xray_photon_flux quantity [keV]")
      ("quantity.xray_photon_flux.num_samples",
       boost::program_options::value<unsigned>(&_xray_photon_flux_num_samples)->default_value(
         _xray_photon_flux_num_samples),
       "Number of samples for the energy integration of the xray_photon_flux quantity")
      ("quantity.luminosity_weighted_temperature.min_energy",
       boost::program_options::value<math::scalar>(&_luminosity_weighted_temp_min_energy)->default_value(
         _luminosity_weighted_temp_min_energy),
       "Start of energy integration range for the luminosity_weighted_temperature quantity [keV]")
      ("quantity.luminosity_weighted_temperature.max_energy",
       boost::program_options::value<math::scalar>(&_luminosity_weighted_temp_max_energy)->default_value(
         _luminosity_weighted_temp_max_energy),
       "End of energy integration range for the luminosity_weighted_temperature quantity [keV]")
      ("quantity.luminosity_weighted_metallicity.min_energy",
       boost::program_options::value<math::scalar>(&_luminosity_weighted_metallicity_min_energy)->default_value(
         _luminosity_weighted_metallicity_min_energy),
       "Start of energy integration range for the luminosity_weighted_metallicity quantity [keV]")
      ("quantity.luminosity_weighted_metallicity.max_energy",
       boost::program_options::value<math::scalar>(&_luminosity_weighted_metallicity_max_energy)->default_value(
         _luminosity_weighted_metallicity_max_energy),
       "End of energy integration range for the luminosity_weighted_metallicity quantity [keV]")
      ("quantity.dm.particle_mass",
       boost::program_options::value<math::scalar>(&_dm_particle_mass)->default_value(_dm_particle_mass),
       "The mass of a dark matter particle [10^10 M_sun/h]");
}

std::unique_ptr<reconstruction_quantity::quantity>
quantity_command_line_parser::create_quantity(const illcrawl_app& app) const
{
  if(_quantity_selection == "chandra_count_rate")
  {
    return std::unique_ptr<reconstruction_quantity::chandra_xray_total_count_rate>{
      new reconstruction_quantity::chandra_xray_total_count_rate{
            &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance()
      }
    };
  }
  else if(_quantity_selection == "chandra_spectral_count_rate")
  {
    assert_greater_zero(this->_chandra_spectral_count_rate_energy, "Energy must be > 0.");
    assert_greater_zero(this->_chandra_spectral_count_rate_energy_bin_width, "Energy bin width must be > 0.");

    return std::unique_ptr<reconstruction_quantity::chandra_xray_spectral_count_rate>{
      new reconstruction_quantity::chandra_xray_spectral_count_rate {
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            this->_chandra_spectral_count_rate_energy,
            this->_chandra_spectral_count_rate_energy_bin_width
      }
    };
  }
  else if(_quantity_selection == "xray_flux")
  {
    assert_greater_zero(this->_xray_flux_num_samples, "Number of integration samples must be > 0");
    assert_valid_integration_range(_xray_flux_min_energy, _xray_flux_max_energy);

    return std::unique_ptr<reconstruction_quantity::xray_flux>{
      new reconstruction_quantity::xray_flux{
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            _xray_flux_min_energy,
            _xray_flux_max_energy,
            _xray_flux_num_samples
      }
    };
  }
  else if(_quantity_selection == "xray_photon_flux")
  {
    assert_greater_zero(this->_xray_photon_flux_num_samples, "Number of integration samples must be > 0");
    assert_greater_zero(this->_xray_photon_flux_min_energy, "Minimum integration energy for photon fluxes must be > 0");
    assert_valid_integration_range(_xray_photon_flux_min_energy, _xray_photon_flux_max_energy);

    return std::unique_ptr<reconstruction_quantity::xray_photon_flux>{
      new reconstruction_quantity::xray_photon_flux{
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            _xray_photon_flux_min_energy,
            _xray_photon_flux_max_energy,
            _xray_photon_flux_num_samples
      }
    };
  }
  else if(_quantity_selection == "xray_spectral_flux")
  {
    assert_greater_zero(this->_xray_spectral_flux_energy, "Energy must be > 0.");
    assert_greater_zero(this->_xray_spectral_flux_energy_bin_width, "Energy bin width must be > 0.");

    return std::unique_ptr<reconstruction_quantity::xray_spectral_flux>{
      new reconstruction_quantity::xray_spectral_flux{
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            _xray_spectral_flux_energy,
            _xray_spectral_flux_energy_bin_width
      }
    };
  }
  else if(_quantity_selection == "mean_temperature")
  {
    return std::unique_ptr<reconstruction_quantity::mean_temperature>{
      new reconstruction_quantity::mean_temperature{
        &(app.get_data_loader()),
            app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "mean_density")
  {
    return std::unique_ptr<reconstruction_quantity::mean_density>{
      new reconstruction_quantity::mean_density{
        &(app.get_data_loader()),
            app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "mass")
  {
    return std::unique_ptr<reconstruction_quantity::mass>{
      new reconstruction_quantity::mass{
        &(app.get_data_loader()),
            app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "potential")
  {
    return std::unique_ptr<reconstruction_quantity::potential>{
      new reconstruction_quantity::potential{
        &(app.get_data_loader()),
            app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "luminosity_weighted_temperature")
  {
    assert_valid_integration_range(_luminosity_weighted_temp_min_energy,
                                   _luminosity_weighted_temp_max_energy);

    return std::unique_ptr<reconstruction_quantity::luminosity_weighted_temperature>{
      new reconstruction_quantity::luminosity_weighted_temperature{
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            _luminosity_weighted_temp_min_energy,
            _luminosity_weighted_temp_max_energy
      }
    };
  }
  else if(_quantity_selection == "metallicity")
  {
    return std::unique_ptr<reconstruction_quantity::metallicity>{
      new reconstruction_quantity::metallicity{
        &(app.get_data_loader()),
        app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "dm_mean_density")
  {
    assert_greater_zero(this->_dm_particle_mass, "Dark matter particle mass must be > 0");

    return std::unique_ptr<reconstruction_quantity::dm_density>{
      new reconstruction_quantity::dm_density {
        &(app.get_data_loader()),
        app.get_unit_converter(),
        _dm_particle_mass
      }
    };
  }
  else if(_quantity_selection == "dm_mass")
  {
    assert_greater_zero(this->_dm_particle_mass, "Dark matter particle mass must be > 0");

    return std::unique_ptr<reconstruction_quantity::dm_mass>{
      new reconstruction_quantity::dm_mass {
        &(app.get_data_loader()),
        app.get_unit_converter(),
        _dm_particle_mass
      }
    };
  }
  else if(_quantity_selection == "stellar_mean_density")
  {
    return std::unique_ptr<reconstruction_quantity::stellar_density>{
      new reconstruction_quantity::stellar_density{
        &(app.get_data_loader()),
        app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "stellar_mass")
  {
    return std::unique_ptr<reconstruction_quantity::stellar_mass>{
      new reconstruction_quantity::stellar_mass{
        &(app.get_data_loader()),
        app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "projected_density")
  {
    return std::unique_ptr<reconstruction_quantity::projected_density>{
      new reconstruction_quantity::projected_density{
        &(app.get_data_loader()),
        app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "density_weighted_metallicity")
  {
    return std::unique_ptr<reconstruction_quantity::density_weighted_metallicity>{
      new reconstruction_quantity::density_weighted_metallicity{
        &(app.get_data_loader()),
        app.get_unit_converter()
      }
    };
  }
  else if(_quantity_selection == "luminosity_weighted_metallicity")
  {
    assert_valid_integration_range(_luminosity_weighted_metallicity_min_energy,
                                   _luminosity_weighted_metallicity_max_energy);

    return std::unique_ptr<reconstruction_quantity::luminosity_weighted_metallicity>{
      new reconstruction_quantity::luminosity_weighted_metallicity{
        &(app.get_data_loader()),
            app.get_unit_converter(),
            app.get_environment().get_compute_device(),
            app.get_redshift(),
            app.get_luminosity_distance(),
            _luminosity_weighted_metallicity_min_energy,
            _luminosity_weighted_metallicity_max_energy
      }
    };
  }
  else
    throw std::invalid_argument("Unknwon quantity: " + _quantity_selection);
}

void
quantity_command_line_parser::save_configuration_to_fits_header(util::fits_header* header) const
{
  header->add_entry("quantity", _quantity_selection, "The rendered quantity");

  header->add_entry("quantity.xray_flux.min_energy",
                    _xray_flux_min_energy,
                    "xray_flux quantity: Minimum energy for the energy integration [keV]");
  header->add_entry("quantity.xray_flux.max_energy",
                    _xray_flux_max_energy,
                    "xray_flux quantity: Maximum energy for the energy integration [keV]");
  header->add_entry("quantity.xray_flux.num_samples",
                    _xray_flux_num_samples,
                    "xray_flux quantity: Number of samples for the energy integration");

  header->add_entry("quantity.xray_photon_flux.min_energy",
                    _xray_photon_flux_min_energy,
                    "xray_photon_flux quantity: Minimum energy for the energy integration [keV]");
  header->add_entry("quantity.xray_photon_flux.max_energy",
                    _xray_photon_flux_max_energy,
                    "xray_photon_flux quantity: Maximum energy for the energy integration [keV]");
  header->add_entry("quantity.xray_photon_flux.num_samples",
                    _xray_photon_flux_num_samples,
                    "xray_photon_flux quantity: Number of samples for the energy integration");

  header->add_entry("quantity.xray_spectral_flux.energy",
                    _xray_spectral_flux_energy,
                    "xray_spectral_flux quantity: Central energy of the evaluated energy channel [keV]");
  header->add_entry("quantity.xray_spectral_flux.energy_bin_width",
                    _xray_spectral_flux_energy_bin_width,
                    "xray_spectral_flux quantity: Width of the evaluated energy channel [keV]");

  header->add_entry("quantity.chandra_spectral_count_rate.energy",
                    _chandra_spectral_count_rate_energy,
                    "chandra_spectral_count_rate quantity: Central energy of the evaluated energy channel [keV]");
  header->add_entry("quantity.chandra_spectral_count_rate.energy_bin_width",
                    _chandra_spectral_count_rate_energy,
                    "chandra_spectral_count_rate quantity: Width of the evaluated energy channel [keV]");

  header->add_entry("quantity.luminosity_weighted_temp.min_energy",
                    _luminosity_weighted_temp_min_energy,
                    "luminosity_weighted_temperature quantity: Minimum energy for the energy integration [keV]");
  header->add_entry("quantity.luminosity_weighted_temp.max_energy",
                    _luminosity_weighted_temp_max_energy,
                    "luminosity_weighted_temperature quantity: Maximum energy for the energy integration [keV]");
  header->add_entry("quantity.dm.particle_mass",
                    _dm_particle_mass,
                    "The dark matter particle mass [10^10 Msun/h]");
}

void
quantity_command_line_parser::assert_valid_integration_range(math::scalar min,
                                                             math::scalar max) const
{
  assert_greater_equal_zero(min, "Start of integration range must be > 0");
  assert_greater_equal_zero(max, "End of integration range must be > 0");
  assert_greater(max, min,
                 "End of integration range must be > than start of integration range");
}


}
