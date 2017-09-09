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


#ifndef VOLUMETRIC_TOMOGRAPHY_HPP
#define VOLUMETRIC_TOMOGRAPHY_HPP

#include <vector>
#include <memory>
#include "reconstructing_data_crawler.hpp"
#include "work_partitioner.hpp"
#include "camera.hpp"
#include "multi_array.hpp"

namespace illcrawl {


class volumetric_tomography
{
public:

  volumetric_tomography(const camera& cam);

  virtual ~volumetric_tomography(){}

  void set_camera(const camera& cam);

  virtual void create_tomographic_cube(reconstructing_data_crawler& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               math::scalar z_range,
                               util::multi_array<device_scalar>& output);

  const camera& get_camera() const;

protected:
  void create_tomographic_cube(reconstructing_data_crawler& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               std::size_t initial_z_step,
                               std::size_t num_steps,
                               util::multi_array<device_scalar>& output) const;

private:

  camera _cam;
};


class distributed_volumetric_tomography : public volumetric_tomography
{
public:
  distributed_volumetric_tomography(const work_partitioner& partitioner,
                                    const camera& cam);

  virtual void create_tomographic_cube(reconstructing_data_crawler& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               math::scalar z_range,
                               util::multi_array<device_scalar>& local_result) override;

  virtual ~distributed_volumetric_tomography(){}

  const work_partitioner& get_partitioning() const;
private:

  std::unique_ptr<work_partitioner> _partitioner;
};

}

#endif
