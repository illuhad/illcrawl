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

#include "volumetric_integration.hpp"

namespace illcrawl {


volumetric_integration::volumetric_integration(
                       const qcl::device_context_ptr& ctx,
                       const camera& cam)
  : _cam{cam}, _ctx{ctx}
{}

void
volumetric_integration::create_projection(reconstructing_data_crawler& reconstruction,
                                          const reconstruction_quantity::quantity& reconstructed_quantity,
                                          math::scalar z_range,
                                          const integration::tolerance& integration_tolerance,
                                          util::multi_array<device_scalar>& output) const
{

  output = util::multi_array<device_scalar>{_cam.get_num_pixels(0),
      _cam.get_num_pixels(1)};

  std::size_t total_num_pixels = _cam.get_num_pixels(0)
      * _cam.get_num_pixels(1);

  std::fill(output.begin(), output.end(), 0.0f);

  integration::parallel_runge_kutta_fehlberg integration_engine{
    _ctx,
    total_num_pixels
  };

  integration::parallel_pixel_integrand integrand{
    _ctx,
    _cam,
    &reconstruction,
    &reconstructed_quantity
  };

  while(integration_engine.get_num_running_integrators() > 0)
  {
    std::cout << integration_engine.get_num_running_integrators()
              << " integrators are still running.\n";
    integration_engine.advance(integration_tolerance, z_range, integrand);
  }

  // Retrieve results
  _ctx->memcpy_d2h(output.data(),
                   integration_engine.get_integration_state(),
                   total_num_pixels);

  // Finalize results

  device_scalar pixel_area =
      static_cast<device_scalar>(_cam.get_pixel_size() * _cam.get_pixel_size());

  device_scalar length_conversion =
      static_cast<device_scalar>(
        reconstructed_quantity.get_unit_converter().length_conversion_factor());

  device_scalar dA = static_cast<device_scalar>(
        reconstructed_quantity.effective_line_of_sight_integration_dA(
          pixel_area * reconstructed_quantity.get_unit_converter().area_conversion_factor(),
          length_conversion * z_range));

  for(auto it = output.begin(); it != output.end(); ++it)
    (*it) *= length_conversion * dA;
}



}
