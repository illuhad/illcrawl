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

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <boost/program_options.hpp>

#include "work_partitioner.hpp"
#include "illcrawl_app.hpp"
#include "python_plot.hpp"
#include "profile.hpp"
#include "reconstructing_data_crawler.hpp"
#include "uniform_work_partitioner.hpp"

void save_profile(const std::vector<illcrawl::math::scalar>& radii,
                  const std::vector<illcrawl::math::scalar>& profile,
                  const std::string& filename)
{
  assert(radii.size() == profile.size());

  std::ofstream output_file(filename);

  if(output_file.is_open())
  {
    output_file << "# Radius [kpc]\t Quantity" << std::endl;

    for(std::size_t i = 0; i < radii.size(); ++i)
    {
      output_file << radii[i] << "\t" << profile[i] << std::endl;
    }
  }
}

int main(int argc, char** argv)
{
  std::string output_name;
  unsigned num_radii;
  illcrawl::math::scalar sampling_density;
  bool display_plot = false;

  boost::program_options::options_description options;
  options.add_options()
      ("output,o",
       boost::program_options::value<std::string>(&output_name)->default_value("cluster_analysis"),
       "the name of the output plot (without file extension)")
      ("num_radii,n",
       boost::program_options::value<unsigned>(&num_radii)->default_value(100),
       "The number of sampled radii")
      ("mean_sampling_density,s",
       boost::program_options::value<illcrawl::math::scalar>(&sampling_density)->default_value(5.e-4),
       "the mean sampling density in units of 1/(ckpc/h)^3")
      ("display_plot,p", boost::program_options::bool_switch(&display_plot),
             "Automatically display plot (The plot script will "
             "always be created for later use)");;


  illcrawl::quantity_command_line_parser quantity_parser;
  quantity_parser.register_options(options);

  illcrawl::illcrawl_app app{argc, argv, std::cout, options};

  try
  {
    app.parse_command_line();
    std::unique_ptr<illcrawl::reconstruction_quantity::quantity> reconstructed_quantity =
        quantity_parser.create_quantity(app);

    std::string coordinate_id = illcrawl::io::illustris_data_loader::get_coordinate_identifier();
    illcrawl::reconstructing_data_crawler reconstructor{
      app.create_reconstruction_backend(*reconstructed_quantity),
      app.get_environment().get_compute_device(),
      app.get_gas_distribution_volume_cutout(),
      app.get_data_loader().get_dataset(coordinate_id),
      app.get_data_crawling_blocksize()
    };

    // Create profile object
    illcrawl::analysis::distributed_mc_radial_profile profile{
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
      illcrawl::python_plot::figure2d result_plot{output_name};

      result_plot.plot(profile.get_proper_profile_radii(), profile_data);
      result_plot.set_x_label("$r$ [kpc]");
      result_plot.save();
      result_plot.show();

      if(display_plot)
      {
        result_plot.generate();
      }
      else
      {
        result_plot.generate_script();
      }

      save_profile(profile.get_proper_profile_radii(), profile_data, output_name+".dat");
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
}

