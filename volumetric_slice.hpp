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

#ifndef VOLUMETRIC_SLICE_HPP
#define VOLUMETRIC_SLICE_HPP

#include <vector>
#include <algorithm>
#include "multi_array.hpp"
#include "camera.hpp"
#include "reconstructing_data_crawler.hpp"

namespace illcrawl {

class volumetric_slice
{
public:

  volumetric_slice(const camera& cam);

  void create_slice(reconstructing_data_crawler& reconstruction,
           const reconstruction_quantity::quantity& reconstructed_quantity,
           util::multi_array<device_scalar>& output,
           std::size_t num_additional_samples = 0) const;

private:


  camera _cam;
};

}

#endif
