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


#include <iostream>

#include "async_io.hpp"
#include "fits.hpp"
#include "hdf5_io.hpp"
#include "particle_distribution.hpp"
#include "qcl.hpp"
#include "reconstruction.hpp"
#include "volumetric_reconstruction.hpp"
#include "environment.hpp"
#include "master_ostream.hpp"
#include "tree_ostream.hpp"
#include "partitioner.hpp"
#include "animation.hpp"
#include "spectrum.hpp"
#include "profile.hpp"

#include "illcrawl.hpp"


using illcrawl::math::scalar;
using illcrawl::device_scalar;
using render_result = illcrawl::util::multi_array<device_scalar>;



struct xray_spectrum_options
{
  illcrawl::math::scalar min_energy;
  illcrawl::math::scalar max_energy;
  std::size_t num_energies;
};

struct command_line_options
{
  // General options
  std::string output_file;
  std::string render_target;
  // Xray spectrum options
  xray_spectrum_options xray_opts;
  // Chandra spectrum options
  std::size_t chandra_spectrum_num_energies;
  // Animation options
  std::size_t animation_num_frames;
  scalar animation_phi_speed;
  // Tomography options
  std::size_t tomography_num_slices;
  // integration
  scalar absolute_tolerance;
  scalar relative_tolerance;
  // Resolution
  std::size_t resolution_x;
  std::size_t resolution_y;
};

template<class Partitioner>
void save_distributed_fits(const Partitioner& partitioning,
                           const std::string& filename,
                           const render_result& data)
{

  illcrawl::util::distributed_fits_slices<Partitioner, device_scalar>
      result_file
  {
    partitioning,
    filename
  };

  result_file.save(data);
}

template<class Spectrum_quantity_creator,
         class Volumetric_reconstructor,
         class Tolerance_type>
void create_spectrum(const illcrawl::illcrawl_app& app,
                     const Spectrum_quantity_creator& quantity_gen,
                     Volumetric_reconstructor& reconstructor,
                     const illcrawl::camera& cam,
                     const Tolerance_type& tol,
                     std::size_t num_energies,
                     const std::string& result_filename,
                     render_result& result)
{
  illcrawl::spectrum::spectrum_generator
  <
      Volumetric_reconstructor,
      illcrawl::uniform_work_partitioner
  > spectrum{
    app.get_environment().get_compute_device(),
    illcrawl::uniform_work_partitioner{app.get_environment().get_communicator()},
    reconstructor
  };

  spectrum(cam, tol, app.get_recommended_integration_depth(), quantity_gen, num_energies, result);
  save_distributed_fits(spectrum.get_partitioning(), result_filename, result);
}

template<class Tolerance_type, class Volumetric_reconstructor>
void run(const illcrawl::illcrawl_app& app,
         const command_line_options& options,
         const Tolerance_type& tol,
         Volumetric_reconstructor& reconstructor,
         const std::unique_ptr<illcrawl::reconstruction_quantity::quantity>& quantity)
{
  render_result result;
  illcrawl::camera cam = app.create_distribution_centered_camera(options.resolution_x, options.resolution_y);

  if(options.render_target == "image")
  {
    app.output_stream() << "Starting volumetric/integrative projection..." << std::endl;

    illcrawl::volumetric_integration<Volumetric_reconstructor> integrator{
      app.get_environment().get_compute_device(),
      cam
    };

    integrator.create_projection(reconstructor,
                                 *quantity,
                                 app.get_recommended_integration_depth(),
                                 tol, result);

    illcrawl::util::fits<device_scalar> result_file{options.output_file};
    result_file.save(result);
  }
  else if(options.render_target == "xray_spectrum")
  {
    app.output_stream() << "Starting calculation of xray spectrum "
                                "(possibly specified quantity will be ignored)"
                             << std::endl;

    illcrawl::spectrum::xray_spectrum_quantity_generator quantity_gen {
      app.get_environment().get_compute_device(),
      app.get_unit_converter(),
      &(app.get_data_loader()),
      app.get_redshift(),
      app.get_luminosity_distance(),
      options.xray_opts.min_energy,
      options.xray_opts.max_energy
    };

    create_spectrum(app, quantity_gen, reconstructor, cam, tol,
                    options.xray_opts.num_energies, options.output_file, result);

  }
  else if(options.render_target == "chandra_spectrum")
  {
    app.output_stream() << "Starting calculation of chandra spectrum "
                                "(possibly specified quantity will be ignored)"
                             << std::endl;

    illcrawl::spectrum::chandra_spectrum_quantity_generator quantity_gen {
      app.get_environment().get_compute_device(),
      app.get_unit_converter(),
      &(app.get_data_loader()),
      app.get_redshift(),
      app.get_luminosity_distance()
    };

    create_spectrum(app, quantity_gen, reconstructor, cam, tol,
                    options.chandra_spectrum_num_energies,
                    options.output_file, result);
  }
  else if(options.render_target == "animation")
  {
    app.output_stream() << "Starting animation..." << std::endl;

    if(options.animation_phi_speed <= 0)
      throw std::invalid_argument{"animation.phi_speed must be > 0."};

    illcrawl::camera_movement::dual_axis_rotation_around_point
    camera_mover{
      app.get_gas_distribution_center(),
      illcrawl::math::vector3{{0,1,0}}, // initial phi axis
      illcrawl::math::vector3{{1,0,0}}, // theta axis
      cam,
      options.animation_phi_speed * 360.0, // phi range
      360.0  // theta range
    };


    illcrawl::frame_rendering::integrated_projection
    <
      Volumetric_reconstructor,
      Tolerance_type
    > frame_renderer {
      app.get_environment().get_compute_device(),
      *quantity, // reconstructed quantity
      app.get_recommended_integration_depth(), // integration depth
      tol, // integration tolerance
      reconstructor // reconstruction engine
    };

    illcrawl::distributed_animation<illcrawl::uniform_work_partitioner>
    animation{
      illcrawl::uniform_work_partitioner{app.get_environment().get_communicator()},
      frame_renderer,
      camera_mover,
      cam
    };

    animation(options.animation_num_frames, result);

    save_distributed_fits(animation.get_partitioning(), options.output_file, result);
  }
  else if(options.render_target == "tomography")
  {
    app.output_stream() << "Starting tomography..." << std::endl;

    illcrawl::distributed_volumetric_tomography
    <
      Volumetric_reconstructor,
      illcrawl::uniform_work_partitioner
    >
    tomography{
      illcrawl::uniform_work_partitioner{app.get_environment().get_communicator()},
      cam
    };

    tomography.create_tomographic_cube(reconstructor,
                                       *quantity,
                                       app.get_recommended_integration_depth(),
                                       result);

    save_distributed_fits(tomography.get_partitioning(), options.output_file, result);
  }
  else
    throw std::runtime_error("Invalid render target: "+options.render_target);
}

int main(int argc, char** argv)
{
  command_line_options cmd_options;

  boost::program_options::options_description options;
  options.add_options()
      ("output,o",
       boost::program_options::value<std::string>(&cmd_options.output_file)->default_value("illcrawl_render.fits"),
       "the output fits file")
      ("render_target,t",
       boost::program_options::value<std::string>(&cmd_options.render_target)->required(),
       "the target type of the render. Can be xray_spectrum,chandra_spectrum,animation,image,tomography.")
      ("xray_spectrum.min_energy",
       boost::program_options::value<scalar>(&cmd_options.xray_opts.min_energy)->default_value(0.1),
       "minimum energy for X-Ray spectra [keV]")
      ("xray_spectrum.max_energy",
       boost::program_options::value<scalar>(&cmd_options.xray_opts.max_energy)->default_value(10.0),
       "maximum energy for X-Ray spectra [keV]")
      ("xray_spectrum.num_energies",
       boost::program_options::value<std::size_t>(&cmd_options.xray_opts.num_energies)->default_value(100),
       "Number of sampled energies for X-ray spectra")
      ("chandra_spectrum.num_energies",
       boost::program_options::value<std::size_t>(&cmd_options.chandra_spectrum_num_energies)->default_value(100),
       "Number of sampled energies for chandra spectra")
      ("animation.num_frames",
       boost::program_options::value<std::size_t>(&cmd_options.animation_num_frames)->default_value(100),
       "Number of frames for animations")
      ("animation.phi_speed",
       boost::program_options::value<scalar>(&cmd_options.animation_phi_speed)->default_value(3.0),
       "Speed of the rotation around the phi axis relative to the rotation speed around the theta axis")
      ("tomography.num_slices",
       boost::program_options::value<std::size_t>(&cmd_options.tomography_num_slices)->default_value(100),
       "Number of slices for tomographies")
      ("integration.absolute_tolerance",
       boost::program_options::value<scalar>(&cmd_options.absolute_tolerance)->default_value(0.0),
       "Absolute tolerance for line-of-sight integration (mutually exclusive with integration.relative_tolerance)")
      ("integration.relative_tolerance",
       boost::program_options::value<scalar>(&cmd_options.relative_tolerance)->default_value(1.e-2),
       "Relative tolerance for line-of-sight integration (mutually exclusive with integration.absolute_tolerance)")
      ("output.resolution.x",
       boost::program_options::value<std::size_t>(&cmd_options.resolution_x)->default_value(1024),
       "Number of pixels in x direction")
      ("output.resolution.y",
       boost::program_options::value<std::size_t>(&cmd_options.resolution_y)->default_value(1024),
       "Number of pixels in y direction");

  illcrawl::quantity_command_line_parser quantity_parser;
  quantity_parser.register_options(options);

  illcrawl::illcrawl_app app{argc, argv, std::cout, options};

  try
  {
    app.parse_command_line();

    std::unique_ptr<illcrawl::reconstruction_quantity::quantity> reconstruction_quantity =
        quantity_parser.create_quantity(app);

    illcrawl::integration::absolute_tolerance<scalar> abs_tol{cmd_options.absolute_tolerance};
    illcrawl::integration::relative_tolerance<scalar> rel_tol{cmd_options.relative_tolerance};

    if(cmd_options.absolute_tolerance > 0.0 && cmd_options.relative_tolerance > 0.0)
      throw std::invalid_argument{"Absolute and relative tolerances cannot both be set; they are mutually exclusive."};
    if(cmd_options.absolute_tolerance <= 0.0 && cmd_options.relative_tolerance <= 0.0)
      throw std::invalid_argument{"Absolute and relative tolerances cannot both be 0 - exactly one of them must be greater than 0."};

    const std::size_t blocksize = 40000000;

    illcrawl::volumetric_nn8_reconstruction reconstructor{
      app.get_environment().get_compute_device(),
      app.get_gas_distribution_volume_cutout(),
      app.get_data_loader().get_coordinates(),
      app.get_data_loader().get_smoothing_length(),
      blocksize
    };

    if(cmd_options.absolute_tolerance > 0.0)
      run(app, cmd_options, abs_tol, reconstructor, reconstruction_quantity);
    else
      run(app, cmd_options, rel_tol, reconstructor, reconstruction_quantity);

  }
  catch(boost::program_options::error& e)
  {
    app.output_stream() << app.get_command_line_options() << std::endl;
  }
  catch(std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
  }
  catch(...)
  {
    std::cout << "Fatal error occured." << std::endl;
  }


  //illcrawl::volumetric_slice<illcrawl::volumetric_nn8_reconstruction> slice{cam};
  //slice.create_slice(reconstructor, *xray_emission, result, 0);


  return 0;
}
