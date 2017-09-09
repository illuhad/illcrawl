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


#ifndef VOLUMETRIC_INTEGRATION_HPP
#define VOLUMETRIC_INTEGRATION_HPP

#include "reconstructing_data_crawler.hpp"
#include "math.hpp"
#include "quantity.hpp"
#include "integration.hpp"

namespace illcrawl {


class volumetric_integration
{
public:

  volumetric_integration(const qcl::device_context_ptr& ctx,
                         const camera& cam);

  void create_projection(reconstructing_data_crawler& reconstruction,
                         const reconstruction_quantity::quantity& reconstructed_quantity,
                         math::scalar z_range,
                         const integration::tolerance& integration_tolerance,
                         util::multi_array<device_scalar>& output) const;

private:


  camera _cam;
  qcl::device_context_ptr _ctx;
};

}

#endif
