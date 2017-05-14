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


#ifndef PARTICLE_DISTRIBUTION_HPP
#define PARTICLE_DISTRIBUTION_HPP

#include <array>
#include <limits>

#include "async_io.hpp"
#include "math.hpp"

namespace illcrawl {

class particle_distribution
{
public:
  particle_distribution(const H5::DataSet& coordinates)
    : _distribution_center{{0.0,0.0,0.0}}, _distribution_size{{0.0,0.0,0.0}}
  {
    io::async_dataset_streamer<math::scalar> streamer{{coordinates}};

    math::vector3 min_coordinates;
    math::vector3 max_coordinates;

    for(std::size_t i = 0; i < 3; ++i)
    {
      min_coordinates[i] = std::numeric_limits<math::scalar>::max();
      max_coordinates[i] = std::numeric_limits<math::scalar>::min();
    }

    io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();
    auto block_processor =
        [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
    {
      for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
      {
        access.select_dataset(0);
        math::vector3 coordinates;
        for(std::size_t j = 0; j < 3; ++j)
        {
          coordinates[j] = access(current_block, i, j);
          if(coordinates[j] > max_coordinates[j])
            max_coordinates[j] = coordinates[j];
          if(coordinates[j] < min_coordinates[j])
            min_coordinates[j] = coordinates[j];
        }
      }
    };


    io::async_for_each_block(streamer.begin_row_blocks(100000),
                             streamer.end_row_blocks(),
                             block_processor);

    _distribution_center = 0.5 * (min_coordinates + max_coordinates);
    _distribution_size = max_coordinates - min_coordinates;

  }

  const math::vector3& get_distribution_center() const
  {
    return _distribution_center;
  }

  const math::vector3& get_distribution_size() const
  {
    return _distribution_size;
  }

private:
  math::vector3 _distribution_center;
  math::vector3 _distribution_size;
};

}

#endif
