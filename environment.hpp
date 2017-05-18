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

#ifndef ENVIRONMENT
#define ENVIRONMENT

#include "qcl.hpp"

namespace illcrawl {

class environment
{
public:
  environment()
  {
    const cl::Platform& plat =
        _env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});

    _global_ctx =
        _env.create_global_context(plat, CL_DEVICE_TYPE_GPU);

    if (_global_ctx->get_num_devices() == 0)
    {
      throw std::runtime_error{"No OpenCL GPU devices found!"};
    }

    _global_ctx->global_register_source_file("reconstruction.cl",
                                            {"image_tile_based_reconstruction2D"});
    _global_ctx->global_register_source_file("volumetric_nn8_reconstruction.cl",
                                            {"volumetric_nn8_reconstruction",
                                             "finalize_volumetric_nn8_reconstruction"});
    _global_ctx->global_register_source_file("interpolation_tree.cl",
                                            {"tree_interpolation"});

    _global_ctx->global_register_source_file("quantities.cl",
                                            // Kernels inside quantities.cl
                                            {
                                              "luminosity_weighted_temperature",
                                              "xray_emission",
                                              "identity",
                                              "mean_temperature"
                                            });
    _global_ctx->global_register_source_file("integration.cl",
                                            {
                                               "runge_kutta_fehlberg",
                                               "construct_evaluation_points_over_camera_plane",
                                               "gather_integrand_evaluations"
                                            });
    _ctx = _global_ctx->device();
  }

  qcl::device_context_ptr get_compute_device() const
  {
    return _ctx;
  }

  qcl::global_context_ptr get_compute_environment() const
  {
    return _global_ctx;
  }
private:
  qcl::global_context_ptr _global_ctx;
  qcl::device_context_ptr _ctx;
  qcl::environment _env;
};

}

#endif
