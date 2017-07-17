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

#include <string>
#include <iostream>
#define FITS_WITHOUT_MPI
#include "fits.hpp"
#include "cl_types.hpp"
#include "multi_array.hpp"

void usage()
{
  std::cout << "Usage: illcrawl_extract_fits_slice <fits data cube> <slice number> <output fits file>" << std::endl;
}

int main(int argc, char** argv)
{
  if(argc != 4)
  {
    usage();
    return -1;
  }

  std::string input_filename = argv[1];
  std::string output_filename = argv[3];

  try
  {
    int slice_id = std::stoi(std::string{argv[2]});

    illcrawl::util::fits<illcrawl::device_scalar> input_cube_file{input_filename};

    illcrawl::util::multi_array<illcrawl::device_scalar> input_cube;
    input_cube_file.load(input_cube);

    if(input_cube.get_dimension() != 3)
      throw std::runtime_error("Input file is not a 3D fits data cube.");

    if(slice_id < 0 ||
       static_cast<std::size_t>(slice_id) >= input_cube.get_extent_of_dimension(2))
      throw std::runtime_error("Slice number is invalid because it is out of bounds for the input file");

    illcrawl::util::multi_array<illcrawl::device_scalar> output_data{
      input_cube.get_extent_of_dimension(0),
      input_cube.get_extent_of_dimension(1)
    };

    for(std::size_t y = 0; y < input_cube.get_extent_of_dimension(1); ++y)
      for(std::size_t x = 0; x < input_cube.get_extent_of_dimension(0); ++x)
      {
        std::size_t idx2 [] = {x,y};
        std::size_t idx3 [] = {x,y,static_cast<std::size_t>(slice_id)};
        output_data[idx2] = input_cube[idx3];
      }

    illcrawl::util::fits<illcrawl::device_scalar> output_file{output_filename};
    output_file.save(output_data);

  }
  catch (std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
  catch(...)
  {
    std::cout << "Unknown error occured." << std::endl;
    return -1;
  }
}

