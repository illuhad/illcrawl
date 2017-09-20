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


#ifndef RECONSTRUCTING_DATA_CRAWLER_HPP
#define RECONSTRUCTING_DATA_CRAWLER_HPP

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <functional>
#include <boost/mpi.hpp>

#include "qcl.hpp"
#include "cl_types.hpp"
#include "math.hpp"
#include "async_io.hpp"
#include "quantity.hpp"
#include "grid.hpp"
#include "multi_array.hpp"
#include "coordinate_system.hpp"

#include "camera.hpp"


#include "environment.hpp"

#include "reconstruction_backend.hpp"

namespace illcrawl {

/// Represents a rectangular volume, defined by
/// center and extent.
class volume_cutout
{
public:
  volume_cutout(const math::vector3& volume_center,
                const math::vector3& volume_extent,
                const math::vector3& periodic_wraparound);

  math::vector3 get_extent(math::scalar border_region_size = 0.0) const;

  math::scalar get_volume() const;

  bool contains_point(const math::vector3& point,
                      math::scalar additional_tolerance = 0.0) const;

  math::vector3 get_bounding_box_min_corner() const;
  math::vector3 get_bounding_box_max_corner() const;

  math::vector3 get_periodic_wraparound_size() const;

private:
  math::vector3 _center;
  math::vector3 _extent;
  math::vector3 _periodic_wraparound_size;

};




class reconstructing_data_crawler
{
public:
  using particle = device_vector4;

  reconstructing_data_crawler(std::unique_ptr<reconstruction_backend> backend,
                           const qcl::device_context_ptr& ctx,
                           const volume_cutout& render_volume,
                           const H5::DataSet& coordinates,
                           std::size_t blocksize = 100000);

  reconstructing_data_crawler& operator=(const reconstructing_data_crawler&) = delete;
  reconstructing_data_crawler(const reconstructing_data_crawler&) = delete;

  /// Sets the reconstruction backend. This also leads
  /// to purging of the reconstruction state.
  void set_backend(std::unique_ptr<reconstruction_backend> backend);

  /// Purges the state of the reconstructor,
  /// making sure that following reconstructions will not reuse
  /// any data from previous reconstructions.
  void purge_state();

  /// Execute the reconstruction
  /// \param evaluation_points The points at which the data shall be
  /// interpolated. This must be a buffer containing elements of type
  /// \c device_vector4. Only the x,y,z components will be used as coordinates.
  /// \param num_evaluation_points The number of evaluation points
  /// \param quantity_to_reconstruct The quantity that shall be estimated
  void run(const cl::Buffer& evaluation_point_coordinate_buffer,
           std::size_t num_evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct);

  /// Execute the reconstruction
  /// \param evaluation_points The points at which the data shall be
  /// interpolated. Only the x,y,z components of the elements will be used as coordinates.
  /// \param quantity_to_reconstruct The quantity that shall be estimated
  void run(const std::vector<device_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct);

  /// \return The number of reconstructed data points
  std::size_t get_num_reconstructed_points() const;

  /// \return An OpenCL data buffer containing the result of the reconstruction
  /// if the reconstruction has already been executed. Otherwise, its content
  /// is undefined.
  const cl::Buffer& get_reconstruction() const;

  /// \return The OpenCL context
  const qcl::device_context_ptr get_context() const;

  /// \return The reconstruction backend
  reconstruction_backend* get_backend();
  /// \return The reconstruction backend
  const reconstruction_backend* get_backend() const;
private:
  /// Executes the quantity transformation
  void run_quantity_transformation(reconstruction_quantity::quantity_transformation& transformation,
                                   std::vector<device_scalar>& transformed_quantity_out) const;

  std::unique_ptr<reconstruction_backend> _backend;
  qcl::device_context_ptr _ctx;
  volume_cutout _render_volume;

  H5::DataSet _coordinate_dataset;

  const std::size_t _blocksize;

  std::size_t _num_reconstructed_points = 0;

  const math::scalar _additional_border_region_size = 100.0;

  std::vector<cl::Buffer> _additional_data_buffers;

  bool _is_first_run = true;
};



}

#endif
