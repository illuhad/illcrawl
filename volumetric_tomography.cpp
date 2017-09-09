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


#include "volumetric_tomography.hpp"
#include "volumetric_slice.hpp"

namespace illcrawl {

/**************** Implementation of volumetric_tomography ****************/

volumetric_tomography::volumetric_tomography(const camera& cam)
  : _cam{cam}
{}


void
volumetric_tomography::set_camera(const camera& cam)
{
  _cam = cam;
}

void
volumetric_tomography::create_tomographic_cube(reconstructing_data_crawler& reconstruction,
                                               const reconstruction_quantity::quantity& reconstructed_quantity,
                                               math::scalar z_range,
                                               util::multi_array<device_scalar>& output)
{

  std::size_t total_num_pixels_z = static_cast<std::size_t>(z_range / _cam.get_pixel_size());
  if(total_num_pixels_z == 0)
    total_num_pixels_z = 1;

  create_tomographic_cube(reconstruction, reconstructed_quantity, 0, total_num_pixels_z, output);

}

const camera&
volumetric_tomography::get_camera() const
{
  return _cam;
}

void
volumetric_tomography::create_tomographic_cube(reconstructing_data_crawler& reconstruction,
                                               const reconstruction_quantity::quantity& reconstructed_quantity,
                                               std::size_t initial_z_step,
                                               std::size_t num_steps,
                                               util::multi_array<device_scalar>& output) const
{

  if(num_steps == 0)
    return;

  output = util::multi_array<device_scalar>{_cam.get_num_pixels(0),
      _cam.get_num_pixels(1),
      num_steps};

  std::fill(output.begin(), output.end(), 0.0f);

  camera moving_cam = _cam;

  for(std::size_t z = 0; z < num_steps; ++z)
  {
    std::cout << "z = " << z << std::endl;

    moving_cam.set_position(_cam.get_position()
                            + (initial_z_step + z) * _cam.get_pixel_size() * moving_cam.get_look_at());

    util::multi_array<device_scalar> slice_data;
    volumetric_slice slice{moving_cam};
    slice.create_slice(reconstruction, reconstructed_quantity, slice_data);

    assert(slice_data.get_dimension() == 2);
    assert(slice_data.get_extent_of_dimension(0) == _cam.get_num_pixels(0));
    assert(slice_data.get_extent_of_dimension(1) == _cam.get_num_pixels(1));

    for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
      for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
      {
        std::size_t output_idx [] = {x,y,z};
        std::size_t slice_idx [] = {x,y};
        output[output_idx] = slice_data[slice_idx];
      }
  }
}

/*************** Implementation of distributed_volumetric_tomography **************/


distributed_volumetric_tomography::distributed_volumetric_tomography(
    const work_partitioner& partitioner,
    const camera& cam)
  : volumetric_tomography{cam},
    _partitioner{std::move(partitioner.clone())}
{}

void
distributed_volumetric_tomography::create_tomographic_cube(
    reconstructing_data_crawler& reconstruction,
    const reconstruction_quantity::quantity& reconstructed_quantity,
    math::scalar z_range,
    util::multi_array<device_scalar>& local_result)
{

  std::size_t total_num_pixels_z = static_cast<std::size_t>(z_range /
                                                            this->get_camera().get_pixel_size());
  if(total_num_pixels_z == 0)
    total_num_pixels_z = 1;

  _partitioner->run(total_num_pixels_z);

  volumetric_tomography::create_tomographic_cube(
        reconstruction,
        reconstructed_quantity,
        _partitioner->own_begin(),
        _partitioner->own_end() - _partitioner->own_begin(),
        local_result);

}


const work_partitioner&
distributed_volumetric_tomography::get_partitioning() const
{
  return *_partitioner;
}



}
