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

#ifndef RECONSTRUCTION_BACKEND_HPP
#define RECONSTRUCTION_BACKEND_HPP

#include <vector>
#include <string>

#include "qcl.hpp"
#include "cl_types.hpp"
#include "particle_grid.hpp"
#include "async_io.hpp"

namespace illcrawl {


class reconstruction_backend
{
public:
  using particle = device_vector4;

  virtual std::vector<H5::DataSet> get_required_additional_datasets() const = 0;

  virtual const cl::Buffer& retrieve_results() = 0;

  virtual std::string get_backend_name() const = 0;

  virtual void init_backend(std::size_t blocksize) = 0;

  virtual void setup_particles(const std::vector<particle>& particles,
                               const std::vector<cl::Buffer>& additional_dataset) = 0;

  virtual void setup_evaluation_points(const cl::Buffer& evaluation_points,
                                       std::size_t num_points) = 0;

  virtual void run() = 0;

  virtual ~reconstruction_backend(){}

};

}

#endif
