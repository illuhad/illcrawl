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

#include "volumetric_slice.hpp"

namespace illcrawl {


volumetric_slice::volumetric_slice(const camera& cam)
  : _cam{cam}
{
}

void
volumetric_slice::create_slice(reconstructing_data_crawler& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               util::multi_array<device_scalar>& output,
                               std::size_t num_additional_samples) const
{

  output = util::multi_array<device_scalar>{_cam.get_num_pixels(0),
      _cam.get_num_pixels(1)};
  std::fill(output.begin(), output.end(), 0.0f);

  std::size_t total_num_pixels = _cam.get_num_pixels(0) *
      _cam.get_num_pixels(1);

  std::random_device rd;
  std::mt19937 random(rd());
  std::uniform_real_distribution<math::scalar> uniform(-0.5, 0.5);

  std::size_t samples_per_pixel = num_additional_samples + 1;

  std::vector<device_vector4> evaluation_points(samples_per_pixel * total_num_pixels);

  for(std::size_t i = 0; i < samples_per_pixel; ++i)
  {
    for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
    {
      for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
      {
        if(i == 0)
        {
          math::vector3 pixel_coord = _cam.get_pixel_coordinate(x,y);
          evaluation_points[y * _cam.get_num_pixels(0) + x] =
              math::to_device_vector4(pixel_coord);
        }
        else
        {
          math::scalar sampled_x = static_cast<math::scalar>(x)+uniform(random);
          math::scalar sampled_y = static_cast<math::scalar>(y)+uniform(random);
          math::vector3 coord = _cam.get_pixel_coordinate(sampled_x, sampled_y);

          evaluation_points[y * _cam.get_num_pixels(0) + x + i * total_num_pixels] =
              math::to_device_vector4(coord);
        }
      }
    }
  }

  reconstruction.run(evaluation_points, reconstructed_quantity);

  // Retrieve results
  std::vector<device_scalar> result_buffer(samples_per_pixel * total_num_pixels);
  reconstruction.get_context()->memcpy_d2h(result_buffer.data(),
                                           reconstruction.get_reconstruction(),
                                           samples_per_pixel * total_num_pixels);


  math::scalar pixel_volume = _cam.get_pixel_size()
      * _cam.get_pixel_size()
      * _cam.get_pixel_size();
  math::scalar dV = reconstructed_quantity.effective_volume_integration_dV(
        pixel_volume * reconstructed_quantity.get_unit_converter().volume_conversion_factor(),
        pixel_volume * reconstructed_quantity.get_unit_converter().volume_conversion_factor());

  for(std::size_t i = 0; i < samples_per_pixel; ++i)
  {
    std::size_t offset = i * total_num_pixels;
    for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
      for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
      {
        std::size_t idx [] = {x,y};
        math::scalar contribution =
            result_buffer[y * _cam.get_num_pixels(0) + x + offset];
        output[idx] += dV * contribution / static_cast<math::scalar>(samples_per_pixel);
      }
  }
}


}
