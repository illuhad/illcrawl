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

#ifndef TYPES_HPP
#define TYPES_HPP

#include "qcl.hpp"
#include <boost/compute.hpp>

namespace illcrawl {

using device_scalar  = cl_float;
using device_vector2 = cl_float2;
using device_vector3 = cl_float3;
using device_vector4 = cl_float4;
using device_vector8 = cl_float8;

using boost_device_vector2 = boost::compute::float2_;
using boost_device_vector4 = boost::compute::float4_;
using boost_device_vector8 = boost::compute::float8_;

}

#endif
