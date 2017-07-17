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
#include <algorithm>
#include <iostream>
#include <cmath>
#define FITS_WITHOUT_MPI
#include "fits.hpp"
#include "multi_array.hpp"
#include "cl_types.hpp"
#include "python_plot.hpp"

void usage()
{
  std::cout << "Usage: illcrawl_sum_pixels <2D or 3D fits file>" << std::endl;
}

int main(int argc, char** argv)
{
  if(argc != 2)
  {
    usage();
    return -1;
  }

  std::string filename = argv[1];

  try
  {
    illcrawl::util::fits<illcrawl::device_scalar> fits_file{filename};

    illcrawl::util::multi_array<illcrawl::device_scalar> data;
    fits_file.load(data);

    std::string plot_name = "pixel_sum";

    if(data.get_dimension() == 2)
    {
      illcrawl::math::scalar sum = 0.0;

      for(std::size_t y = 0; y < data.get_extent_of_dimension(1); ++y)
        for(std::size_t x = 0; x < data.get_extent_of_dimension(0); ++x)
        {
          std::size_t idx[] = {x,y};
          illcrawl::math::scalar current_val =
              static_cast<illcrawl::math::scalar>(data[idx]);

          if(std::isfinite(current_val))
            sum += current_val;
        }
      std::cout << sum << std::endl;
    }
    else if(data.get_dimension() == 3)
    {
      std::vector<illcrawl::math::scalar> sums(data.get_extent_of_dimension(2), 0.0);
      std::vector<illcrawl::math::scalar> slice_ids(data.get_extent_of_dimension(2), 0.0);

      for(std::size_t slice = 0; slice < data.get_extent_of_dimension(2); ++slice)
      {
        slice_ids[slice] = static_cast<illcrawl::math::scalar>(slice);

        for(std::size_t y = 0; y < data.get_extent_of_dimension(1); ++y)
        {
          for(std::size_t x = 0; x < data.get_extent_of_dimension(0); ++x)
          {
            std::size_t idx [] = {x, y, slice};

            illcrawl::device_scalar current_val = data[idx];

            if(std::isfinite(current_val))
            {
              sums[slice] += static_cast<illcrawl::math::scalar>(current_val);
            }

          }
        }

        std::cout << slice << "\t" << sums[slice] << std::endl;
      }

      illcrawl::python_plot::figure2d result_plot{plot_name};

      result_plot.plot(slice_ids, sums);

      result_plot.save();
      result_plot.show();
      result_plot.generate();
    }
    else
      throw std::invalid_argument{"Invalid dimension of fits file: "
                                 +std::to_string(data.get_dimension())};
  }
  catch(std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
  }
  catch(...)
  {
    std::cout << "Fatal error" << std::endl;
  }
}
