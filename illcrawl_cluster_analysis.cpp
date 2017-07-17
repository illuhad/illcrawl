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
#include <stdexcept>
#include <vector>
#include <boost/program_options.hpp>

#include "partitioner.hpp"
#include "illcrawl.hpp"
#include "python_plot.hpp"
#include "profile.hpp"
#include "volumetric_reconstruction.hpp"

int main(int argc, char** argv)
{
  std::string plot_name;
  unsigned num_radii;
  illcrawl::math::scalar sampling_density;

  boost::program_options::options_description options;
  options.add_options()
      ("output,o",
       boost::program_options::value<std::string>(&plot_name)->default_value("cluster_analysis"),
       "the name of the output plot (without file extension)")
      ("num_radii,n",
       boost::program_options::value<unsigned>(&num_radii)->default_value(100),
       "The number of sampled radii")
      ("mean_sampling_density,s",
       boost::program_options::value<illcrawl::math::scalar>(&sampling_density)->default_value(5.e-4),
       "the mean sampling density in units of 1/(ckpc/h)^3");


  illcrawl::quantity_command_line_parser quantity_parser;
  quantity_parser.register_options(options);

  illcrawl::illcrawl_app app{argc, argv, std::cout, options};

  try
  {
    app.parse_command_line();
    std::unique_ptr<illcrawl::reconstruction_quantity::quantity> reconstructed_quantity =
        quantity_parser.create_quantity(app);

    const std::size_t blocksize = 40000000;

    illcrawl::volumetric_nn8_reconstruction reconstructor{
      app.get_environment().get_compute_device(),
      app.get_gas_distribution_volume_cutout(),
      app.get_data_loader().get_coordinates(),
      app.get_data_loader().get_smoothing_length(),
      blocksize
    };

    // Create profile object
    illcrawl::analysis::distributed_mc_radial_profile<
        illcrawl::uniform_work_partitioner,
        illcrawl::volumetric_nn8_reconstruction
    >
    profile{
      app.get_environment().get_compute_device(),
      illcrawl::uniform_work_partitioner{app.get_environment().get_communicator()},
      app.get_gas_distribution_center(),
      0.5 * app.get_gas_distribution_size()[0],
      num_radii,
    };

    // Calculate profile data
    std::vector<illcrawl::math::scalar> profile_data;
    profile(reconstructor,
            *reconstructed_quantity,
            sampling_density,
            profile_data,
            app.get_environment().get_master_rank());
    // Wait for all processes to finish
    app.get_environment().get_communicator().barrier();

    // Create plot on master process
    if(app.get_environment().get_communicator().rank() ==
       app.get_environment().get_master_rank())
    {
      illcrawl::python_plot::figure2d result_plot{plot_name};

      result_plot.plot(profile.get_profile_radii(), profile_data);
      result_plot.set_x_label("$r$ [ckpc/h]");
      result_plot.save();
      result_plot.show();
      result_plot.generate();
    }
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
};

