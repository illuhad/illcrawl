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
#include "projective_smoothing_reconstruction.hpp"
#include "projection.hpp"

namespace illcrawl {

projection::projection(const qcl::device_context_ptr& ctx,
                       const camera& cam)
  : _ctx{ctx},
    _cam{cam}
{
}

void
projection::project_by_numerical_integration(reconstructing_data_crawler& reconstruction,
                                          const reconstruction_quantity::quantity& reconstructed_quantity,
                                          math::scalar z_range,
                                          const integration::tolerance& integration_tolerance,
                                          util::multi_array<device_scalar>& output) const
{
  volumetric_integration integrator{_ctx, _cam};
  integrator.create_projection(reconstruction,
                               reconstructed_quantity,
                               z_range,
                               integration_tolerance,
                               output);
}

void
projection::create_projection(reconstructing_data_crawler& reconstruction,
                              const reconstruction_quantity::quantity& reconstructed_quantity,
                              math::scalar z_range,
                              const integration::tolerance& integration_tolerance,
                              util::multi_array<device_scalar>& output) const
{
  if(reconstructed_quantity.is_quantity_baryonic())
  {
    this->project_by_numerical_integration(reconstruction,
                                        reconstructed_quantity,
                                        z_range,
                                        integration_tolerance,
                                        output);
  }
  else
  {
    // Use projective smoothing. In this case, the reconstructor
    // should already be initialized with a projective_smoothing_backend.
    using projective_smoothing_ptr = reconstruction_backends::dm::projective_smoothing*;

    projective_smoothing_ptr backend =
        dynamic_cast<projective_smoothing_ptr>(reconstruction.get_backend());

    if(backend == nullptr)
    {
      // We are not dealing with a reconstructor that has a
      // projective smoothing backend. Fallback
      // to numerical integration
      this->project_by_numerical_integration(reconstruction,
                                          reconstructed_quantity,
                                          z_range,
                                          integration_tolerance,
                                          output);
    }
    else
    {

      backend->set_camera(_cam);
      backend->set_integration_depth(z_range);

      cl::Buffer evaluation_points;
      _cam.generate_pixel_coordinates(_ctx, evaluation_points);

      std::size_t total_num_pixels =
          _cam.get_num_pixels(0)*_cam.get_num_pixels(1);

      reconstruction.run(evaluation_points,
                         total_num_pixels,
                         reconstructed_quantity);

      // retrieve results
      const cl::Buffer& reconstruction_result =
          reconstruction.get_reconstruction();
      output = util::multi_array<device_scalar>{
          _cam.get_num_pixels(0),
          _cam.get_num_pixels(1)
      };

      _ctx->memcpy_d2h(output.data(),
                       reconstruction_result,
                       total_num_pixels);
    }
  }
}

}
