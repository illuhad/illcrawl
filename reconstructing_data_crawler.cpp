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

#include "reconstructing_data_crawler.hpp"

namespace illcrawl {

/*************** Implementation of volume_cutout ********************/

volume_cutout::volume_cutout(const math::vector3& volume_center,
                             const math::vector3& volume_extent,
                             const math::vector3& periodic_wraparound)
  : _center(volume_center),
    _extent(volume_extent),
    _periodic_wraparound_size(periodic_wraparound)
{}

inline math::vector3 volume_cutout::get_extent(math::scalar border_region_size) const
{
  math::vector3 result = _extent;
  for(std::size_t i = 0; i < 3; ++i)
    result[i] += 2 * border_region_size;
  return result;
}

inline math::scalar volume_cutout::get_volume() const
{
  return _extent[0] * _extent[1] * _extent[2];
}

inline bool volume_cutout::contains_point(const math::vector3& point,
                                          math::scalar additional_tolerance) const
{
  for(std::size_t i = 0; i < 3; ++i)
  {
    if(point[i] < (_center[i] - 0.5 * _extent[i] - additional_tolerance))
      return false;
    if(point[i] > (_center[i] + 0.5 * _extent[i] + additional_tolerance))
      return false;
  }
  return true;
}

inline math::vector3 volume_cutout::get_bounding_box_min_corner() const
{
  return _center - 0.5 * _extent;
}

inline math::vector3 volume_cutout::get_bounding_box_max_corner() const
{
  return _center + 0.5 * _extent;
}

inline math::vector3 volume_cutout::get_periodic_wraparound_size() const
{
  return _periodic_wraparound_size;
}

/**************** Implementation of volumetric_reconstructor *****************/

reconstructing_data_crawler::reconstructing_data_crawler(
                         std::unique_ptr<reconstruction_backend> backend,
                         const qcl::device_context_ptr& ctx,
                         const volume_cutout& render_volume,
                         const H5::DataSet& coordinates,
                         std::size_t blocksize)
  : _backend{std::move(backend)},
    _ctx{ctx},
    _render_volume{render_volume},
    _coordinate_dataset{coordinates},
    _blocksize{blocksize}
{
  this->_backend->init_backend(blocksize);
  _additional_data_buffers.resize(this->_backend->get_required_additional_datasets().size());


  for(cl::Buffer& buff : _additional_data_buffers)
    _ctx->create_input_buffer<device_scalar>(buff, _blocksize);

}


void reconstructing_data_crawler::purge_state()
{
  this->_is_first_run = true;
}

void reconstructing_data_crawler::run(
         const cl::Buffer& evaluation_point_coordinate_buffer,
         std::size_t num_evaluation_points,
         const reconstruction_quantity::quantity& quantity_to_reconstruct)
{
  _num_reconstructed_points = num_evaluation_points;

  this->_backend->setup_evaluation_points(evaluation_point_coordinate_buffer,
                                          num_evaluation_points);

  std::vector<H5::DataSet> streamed_quantities{
    _coordinate_dataset
  };

  auto additional_required_datasets = _backend->get_required_additional_datasets();
  for(const H5::DataSet& dataset : additional_required_datasets)
    streamed_quantities.push_back(dataset);

  // This will serve as a buffer for additional datasets that may be required
  // by the reconstruction engine (e.g. smoothing lengths).
  std::vector<std::vector<device_scalar>> additional_required_data(
        additional_required_datasets.size());

  for(const H5::DataSet& dataset : quantity_to_reconstruct.get_required_datasets())
    streamed_quantities.push_back(dataset);

  std::size_t num_input_quantities =
      quantity_to_reconstruct.get_required_datasets().size();
  std::vector<device_scalar> quantity_transformation_input_buffer(num_input_quantities);
  std::vector<device_scalar> transformed_quantity;
  std::vector<particle> particles;

  io::async_dataset_streamer<math::scalar> streamer{streamed_quantities};

  // If we have done a reconstruction previously, and the blocksize is
  // larger than the number of rows, we do not need to call
  // _backend->setup_particles() again, because the correct particles
  // are still present in the backend. This means that we can skip all
  // IO, and can start the reconstruction directly.
  if(!this->_is_first_run && (streamer.get_num_rows() <= _blocksize))
  {
    this->_backend->run();
  }
  else
  {
    reconstruction_quantity::quantity_transformation transformation{
      _ctx,
          quantity_to_reconstruct,
          _blocksize
    };

    io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();
    io::async_for_each_block(streamer.begin_row_blocks(_blocksize),
                             streamer.end_row_blocks(),
                             [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
    {
      // Don't bother if there's nothing to do..
      if(current_block.get_num_available_rows() == 0)
        return;

      transformation.clear();
      particles.clear();

      for(auto& data_buffer : additional_required_data)
        data_buffer.clear();

      for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
      {
        // Correct particle peridicity
        access.select_dataset(0);
        math::vector3 coordinates;
        for(std::size_t j = 0; j < 3; ++j)
          // The static_cast ensures that the rounding errors
          // remain consistent
          coordinates[j] = static_cast<device_scalar>(access(current_block, i, j));

        coordinate_system::correct_periodicity(_render_volume.get_bounding_box_min_corner(),
                                               _render_volume.get_bounding_box_max_corner(),
                                               _render_volume.get_periodic_wraparound_size(),
                                               _additional_border_region_size,
                                               coordinates);

        // Check if particle is included in render volume
        if(_render_volume.contains_point(coordinates))
        {
          // Extract datasets additionally required by the backend
          for(std::size_t additional_dataset_id = 0;
              additional_dataset_id < additional_required_datasets.size();
              ++additional_dataset_id)
          {
            access.select_dataset(1 + additional_dataset_id);
            // Read datasets additionally required by the reconstruction backend
            additional_required_data[additional_dataset_id].push_back(
                  static_cast<device_scalar>(access(current_block, i)));
          }

          // Fill quantity transformation input
          assert(quantity_transformation_input_buffer.size() == num_input_quantities);
          for(std::size_t quantity_dataset_id = 0;
              quantity_dataset_id < num_input_quantities;
              ++quantity_dataset_id)
          {
            std::size_t dataset_id =
                1 + additional_required_datasets.size() + quantity_dataset_id;

            access.select_dataset(dataset_id);
            // Input quantities with dimension > 1 are currently unsupported.
            assert(access.get_dataset_shape(dataset_id).size() == 1
                   || access.get_dataset_shape(dataset_id)[1] == 1);

            quantity_transformation_input_buffer[quantity_dataset_id] =
                static_cast<device_scalar>(access(current_block, i));
          }

          // Add particle data to quantity transformation queue
          transformation.queue_input_quantities(quantity_transformation_input_buffer);
          // Build particle
          particles.push_back(math::to_device_vector4(coordinates));
        }
      }

      // Run quantity transformation
      run_quantity_transformation(transformation, transformed_quantity);

      // Combine particles with their quantity value
      assert(particles.size() == transformed_quantity.size());
      for(std::size_t i = 0; i < particles.size(); ++i)
        particles[i].s[3] = transformed_quantity[i];

      // Copy additionally required datasets to device
      assert(additional_required_data.size() == this->_additional_data_buffers.size());
      for(std::size_t i = 0; i < additional_required_data.size(); ++i)
      {
        _ctx->memcpy_h2d<device_scalar>(_additional_data_buffers[i],
                                        additional_required_data[i].data(),
                                        additional_required_data[i].size());
      }
      // Setup particles
      this->_backend->setup_particles(particles, this->_additional_data_buffers);

      // Run reconstruction
      this->_backend->run();
    });
    this->_is_first_run = false;
  }
}

void reconstructing_data_crawler::run(
         const std::vector<device_vector4>& evaluation_points,
         const reconstruction_quantity::quantity& quantity_to_reconstruct)
{
  cl::Buffer evaluation_point_coordinate_buffer;

  _ctx->create_input_buffer<device_vector4>(evaluation_point_coordinate_buffer, evaluation_points.size());
  _ctx->memcpy_h2d(evaluation_point_coordinate_buffer,
                  evaluation_points.data(),
                  evaluation_points.size());

  this->run(evaluation_point_coordinate_buffer,
            evaluation_points.size(),
            quantity_to_reconstruct);
}

std::size_t reconstructing_data_crawler::get_num_reconstructed_points() const
{
  return this->_num_reconstructed_points;
}

const cl::Buffer&
reconstructing_data_crawler::get_reconstruction() const
{
  return _backend->retrieve_results();
}


const qcl::device_context_ptr
reconstructing_data_crawler::get_context() const
{
  return _ctx;
}

void
reconstructing_data_crawler::run_quantity_transformation(
    reconstruction_quantity::quantity_transformation& transformation,
    std::vector<device_scalar>& transformed_quantity_out) const
{
  transformation.commit_data();
  cl::Event transformation_complete;
  transformation(&transformation_complete);

  cl_int err = transformation_complete.wait();
  qcl::check_cl_error(err, "Error while waiting for the quantity transformation"
                           "to complete.");


  cl::Event transformation_result_retrieved;
  transformation.retrieve_results(&transformation_result_retrieved,
                                  transformed_quantity_out);

  err = transformation_result_retrieved.wait();
  qcl::check_cl_error(err, "Error while retrieving the quantity transformation results.");

}

}
