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


#ifndef VOLUMETRIC_RECONSTRUCTION_HPP
#define VOLUMETRIC_RECONSTRUCTION_HPP

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
#include "interpolation_tree.hpp"
#include "particle_grid.hpp"
#include "camera.hpp"
#include "partitioner.hpp"

#include "integration.hpp"
#include "environment.hpp"

namespace illcrawl {

/// Represents a rectangular volume, defined by
/// center and extent.
struct volume_cutout
{
  math::vector3 center;
  math::vector3 extent;
  math::vector3 periodic_wraparound_size;


  inline math::vector3 get_extent(math::scalar border_region_size = 0.0) const
  {
    math::vector3 result = extent;
    for(std::size_t i = 0; i < 3; ++i)
      result[i] += 2 * border_region_size;
    return result;
  }

  inline math::scalar get_volume() const
  {
    return extent[0] * extent[1] * extent[2];
  }

  inline bool contains_point(const math::vector3& point,
                             math::scalar additional_tolerance = 0.0) const
  {
    for(std::size_t i = 0; i < 3; ++i)
    {
      if(point[i] < (center[i] - 0.5 * extent[i] - additional_tolerance))
        return false;
      if(point[i] > (center[i] + 0.5 * extent[i] + additional_tolerance))
        return false;
    }
    return true;
  }

  inline math::vector3 get_bounding_box_min_corner() const
  {
    return center - 0.5 * extent;
  }

  inline math::vector3 get_bounding_box_max_corner() const
  {
    return center + 0.5 * extent;
  }

  volume_cutout(const math::vector3& volume_center,
                const math::vector3& volume_extent,
                const math::vector3& periodic_wraparound)
    : center(volume_center),
      extent(volume_extent),
      periodic_wraparound_size(periodic_wraparound)
  {}
};

/*
class volumetric_reconstruction_backend
{
public:
  using particle = device_vector4;

  virtual void purge_state() = 0;

  virtual void get_required_additional_datasets(std::vector<H5::DataSet>& datasets) = 0;

  virtual void run(const cl::Buffer& reconstruction_point_coordinates,
                   std::size_t       num_points) = 0;
  virtual std::string get_backend_name() const = 0;

  virtual void init_backend(const cl::Buffer& particles,
                            std::size_t num_particles,
                            std::size_t blocksize) = 0;
};*/

/// Reconstructs the value of a given quantity at a given (arbitrary) set of
/// 3D evaluation points. The values are calculated by a inverse distance interpolation
/// scheme using the 8 nearest neighbors.
class volumetric_nn8_reconstruction
{
public:

  using nearest_neighbor_list = device_vector8;
  using particle = particle_tile_grid::particle;

  const math::scalar smoothing_length_scale_factor = 0.5;
  const math::scalar additional_border_region_size = 100.0;
  const std::size_t local_size1D = 64;
  const std::size_t local_size2D = 16;

  volumetric_nn8_reconstruction(const qcl::device_context_ptr& ctx,
                            const volume_cutout& render_volume,
                            const H5::DataSet& coordinates,
                            const H5::DataSet& smoothing_lengths,
                            std::size_t blocksize = 100000)
    : _ctx{ctx},
      _render_volume{render_volume},
      _coordinates{coordinates},
      _smoothing_lengths{smoothing_lengths},
      _kernel{ctx->get_kernel("volumetric_nn8_reconstruction")},
      _finalization_kernel{ctx->get_kernel("finalize_volumetric_nn8_reconstruction")},
      _num_reconstructed_points{0},
      _blocksize{blocksize},
      _transformed_quantity(blocksize)
  {

  }

  /// Purges the state of the reconstructor,
  /// making sure that following reconstructions will not reuse
  /// any data from previous reconstructions.
  void purge_state()
  {
    this->_grid = nullptr;
  }

  /// Execute the reconstruction
  /// \param evaluation_points The points at which the data shall be
  /// interpolated
  /// \param num_evaluation_points The number of evaluation points
  /// \param quantity_to_reconstruct The quantity that shall be estimated
  void run(const cl::Buffer& evaluation_point_coordinate_buffer,
           std::size_t num_evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct)
  {
    std::vector<H5::DataSet> streamed_quantities{
      _coordinates,
      _smoothing_lengths
    };

    for(const H5::DataSet& quantity : quantity_to_reconstruct.get_required_datasets())
      streamed_quantities.push_back(quantity);

    std::size_t num_input_quantities =
        quantity_to_reconstruct.get_required_datasets().size();
    std::vector<device_scalar> input_quantities(num_input_quantities);

    io::async_dataset_streamer<math::scalar> streamer{streamed_quantities};


    cl::Buffer evaluation_point_weights_buffer;
    cl::Buffer evaluation_point_values_buffer;


    _ctx->create_buffer<nearest_neighbor_list>(evaluation_point_weights_buffer,
                                               CL_MEM_READ_WRITE,
                                               num_evaluation_points);
    _ctx->create_buffer<nearest_neighbor_list>(evaluation_point_values_buffer,
                                               CL_MEM_READ_WRITE,
                                               num_evaluation_points);

    // Copy evaluation points to the GPU
    bool is_first_block = true;

    cl::Event kernel_finished;
    cl_int err;
    if((_grid != nullptr) &&
       (streamer.get_num_rows() <= _blocksize))
    {
      // Reuse existing particles and tiles

      this->launch_reconstruction(num_evaluation_points,
                                  is_first_block,
                                  evaluation_point_coordinate_buffer,
                                  evaluation_point_values_buffer,
                                  evaluation_point_weights_buffer,
                                  &kernel_finished);

      is_first_block = false;
    }
    else
    {

      reconstruction_quantity::quantity_transformation transformation{
        _ctx, quantity_to_reconstruct, _blocksize};


      io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();
      io::async_for_each_block(streamer.begin_row_blocks(_blocksize),
                               streamer.end_row_blocks(),
                               [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
      {
        // Don't bother if there's nothing to do..
        if(current_block.get_num_available_rows() == 0)
          return;

        transformation.clear();
        _filtered_particles.clear();
        _filtered_smoothing_lengths.clear();

        for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
        {
          access.select_dataset(0);
          math::vector3 coordinate;
          for(std::size_t j = 0; j < 3; ++j)
            // The static_cast ensures that the rounding errors
            // remain consistent
            coordinate[j] = static_cast<device_scalar>(access(current_block, i, j));

          coordinate_system::correct_periodicity(_render_volume.get_bounding_box_min_corner(),
                                                 _render_volume.get_bounding_box_max_corner(),
                                                 _render_volume.periodic_wraparound_size,
                                                 additional_border_region_size,
                                                 coordinate);

          if(_render_volume.contains_point(coordinate, additional_border_region_size))
          {
            access.select_dataset(1);
            math::scalar smoothing_length =
                smoothing_length_scale_factor * access(current_block, i);

            for(std::size_t quantity_index = 0;
                quantity_index < num_input_quantities;
                ++quantity_index)
            {
              access.select_dataset(2 + quantity_index);
              input_quantities[quantity_index] =
                  static_cast<device_scalar>(access(current_block, i));
            }

            particle new_particle;
            for(std::size_t j = 0; j < 3; ++j)
              new_particle.s[j] = static_cast<device_scalar>(coordinate[j]);

            _filtered_particles.push_back(new_particle);
            _filtered_smoothing_lengths.push_back(static_cast<device_scalar>(smoothing_length));

            transformation.queue_input_quantities(input_quantities);
          }
        }

        assert(_filtered_particles.size() == _filtered_smoothing_lengths.size());

        // Make sure we actually have work to do for
        // this block before involving the GPU
        if(_filtered_particles.size() > 0)
        {
          std::cout << "Processing "
                    << _filtered_particles.size()
                    << " particles between "
                    << current_block.get_available_data_range_begin() << " / "
                    << current_block.get_available_data_range_end() << std::endl;

          transformation.commit_data();
          cl::Event transformation_complete;
          // Transform quantities
          transformation(&transformation_complete);

          err = transformation_complete.wait();
          qcl::check_cl_error(err, "Error while waiting for the quantity transformation"
                                   "to complete.");


          cl::Event transformation_result_retrieved;
          transformation.retrieve_results(&transformation_result_retrieved,
                                          _transformed_quantity);

          err = transformation_result_retrieved.wait();
          qcl::check_cl_error(err, "Error while retrieving the quantity transformation results.");

          assert(_transformed_quantity.size() == _filtered_particles.size());

          for(std::size_t i = 0; i < _filtered_particles.size(); ++i)
            _filtered_particles[i].s[3] = _transformed_quantity[i];

          // Release old memory first by setting _grid to null
          _grid = nullptr;
          this->_grid = std::make_shared<particle_grid>(_ctx,
                                                        _filtered_particles);

          // Wait for the previous kernel to finish, if we are not
          // in the first run
          if(!is_first_block)
          {
            err = kernel_finished.wait();
            qcl::check_cl_error(err, "Error while waiting for the volumetric_nn8_reconstruction"
                                   " kernel to finish.");
          }

          this->launch_reconstruction(num_evaluation_points,
                                      is_first_block,
                                      evaluation_point_coordinate_buffer,
                                      evaluation_point_values_buffer,
                                      evaluation_point_weights_buffer,
                                      &kernel_finished);


          is_first_block = false;

        }

      });

      // Wait for the last kernel to finish
      if(!is_first_block)
      {
        err = kernel_finished.wait();
        qcl::check_cl_error(err, "Error while waiting for the volumetric reconstruction kernel to finish.");
      }

    }
    _num_reconstructed_points = num_evaluation_points;
    _ctx->create_buffer<device_scalar>(_reconstruction_result,
                                       CL_MEM_READ_WRITE,
                                       _num_reconstructed_points);

    qcl::kernel_argument_list finalization_args{_finalization_kernel};
    finalization_args.push(static_cast<cl_int>(_num_reconstructed_points));
    finalization_args.push(evaluation_point_weights_buffer);
    finalization_args.push(evaluation_point_values_buffer);
    finalization_args.push(_reconstruction_result);

    cl::Event finalization_done;
    err =_ctx->get_command_queue().enqueueNDRangeKernel(*_finalization_kernel,
                                                   cl::NullRange,
                                                   cl::NDRange(math::make_multiple_of(local_size1D,
                                                                                      num_evaluation_points)),
                                                   cl::NDRange(local_size1D),
                                                   nullptr,
                                                   &finalization_done);
    qcl::check_cl_error(err, "Could not enqueue finalization kernel.");

    err = finalization_done.wait();
    qcl::check_cl_error(err, "Error while waiting for finalization kernel to finish.");
  }


  void run(const std::vector<device_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct)
  {
    cl::Buffer evaluation_point_coordinate_buffer;

    _ctx->create_input_buffer<device_vector4>(evaluation_point_coordinate_buffer, evaluation_points.size());

    _ctx->memcpy_h2d(evaluation_point_coordinate_buffer,
                    evaluation_points.data(),
                    evaluation_points.size());

    run(evaluation_point_coordinate_buffer, evaluation_points.size(), quantity_to_reconstruct);
  }

  /// \return The number of reconstructed data points
  std::size_t get_num_reconstructed_points() const
  {
    return _num_reconstructed_points;
  }

  /// \return An OpenCL data buffer containing the result of the reconstruction
  /// if the reconstruction has already been executed. Otherwise, its content
  /// is undefined.
  const cl::Buffer& get_reconstruction() const
  {
    return _reconstruction_result;
  }

  /// \return The OpenCL context
  const qcl::device_context_ptr get_context() const
  {
    return _ctx;
  }
private:

  void launch_reconstruction(std::size_t num_evaluation_points,
                             bool is_first_run,
                             const cl::Buffer& evaluation_points,
                             const cl::Buffer& evaluation_points_values,
                             const cl::Buffer& evaluation_points_weights,
                             cl::Event* kernel_finished_event)
  {
    assert(_grid != nullptr);

    auto num_cells = _grid->get_num_grid_cells();
    auto grid_min_corner = _grid->get_grid_min_corner();
    auto cell_sizes = _grid->get_grid_cell_sizes();

    qcl::kernel_argument_list args{_kernel};
    args.push(static_cast<cl_int>(is_first_run));
    args.push(_grid->get_grid_cells_buffer());
    args.push(num_cells);
    args.push(grid_min_corner);
    args.push(cell_sizes);

    args.push(_grid->get_particle_buffer());

    args.push(static_cast<cl_int>(num_evaluation_points));
    args.push(evaluation_points);
    args.push(evaluation_points_weights);
    args.push(evaluation_points_values);

    std::vector<cl::Event> grid_ready = {{_grid->get_grid_ready_event()}};
    cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(*_kernel,
                                                   cl::NullRange,
                                                   cl::NDRange{math::make_multiple_of(local_size1D, num_evaluation_points)},
                                                   cl::NDRange{local_size1D},
                                                   &grid_ready,
                                                   kernel_finished_event);

    qcl::check_cl_error(err, "Could not enqueue volumetric_nn8_reconstruction kernel");


  }


  qcl::device_context_ptr _ctx;
  volume_cutout _render_volume;

  H5::DataSet _coordinates;
  H5::DataSet _smoothing_lengths;

  qcl::kernel_ptr _kernel;
  qcl::kernel_ptr _finalization_kernel;

  std::size_t _num_reconstructed_points;
  cl::Buffer  _reconstruction_result;

  std::shared_ptr<particle_grid> _grid;
  std::size_t _blocksize;

  std::vector<particle> _filtered_particles;
  std::vector<device_scalar> _filtered_smoothing_lengths;
  std::vector<device_scalar> _transformed_quantity;
};

class volumetric_tree_reconstruction
{
public:
  volumetric_tree_reconstruction(const qcl::device_context_ptr& ctx,
                                 const volume_cutout& render_volume,
                                 const H5::DataSet& coordinates,
                                 std::size_t blocksize = 100000,
                                 math::scalar opening_angle = 0.5)
    : _blocksize{blocksize},
      _ctx{ctx},
      _render_volume{render_volume},
      _coordinates{coordinates},
      _tree{nullptr},
      _num_reconstructed_points{0},
      _opening_angle{opening_angle}
  {

  }


  using nearest_neighbor_list = device_vector8;
  using particle = device_vector4;

  /// Purges the state of the reconstructor,
  /// making sure that following reconstructions will not reuse
  /// any data from previous reconstructions.
  void purge_state()
  {
    this->_tree = nullptr;
  }

  void run(const cl::Buffer& evaluation_points_buffer,
           std::size_t num_evaluation_points,
           const reconstruction_quantity::quantity& q)
  {
    _evaluation_points_buffer = evaluation_points_buffer;

    std::vector<H5::DataSet> streamed_data = {{_coordinates}};
    auto required_datasets = q.get_required_datasets();
    for(const H5::DataSet& data : required_datasets)
      streamed_data.push_back(data);

    io::async_dataset_streamer<math::scalar> streamer{streamed_data};

    _ctx->create_buffer<device_scalar>(_reconstruction_result,
                                       CL_MEM_READ_WRITE,
                                       num_evaluation_points);
    _ctx->create_buffer<device_scalar>(_reconstruction_value_sum_state_buffer,
                                       CL_MEM_READ_WRITE,
                                       num_evaluation_points);
    _ctx->create_buffer<device_scalar>(_reconstruction_weight_sum_state_buffer,
                                       CL_MEM_READ_WRITE,
                                       num_evaluation_points);



    if((_tree != nullptr) &&
       (streamer.get_num_rows() <= _blocksize))
    {
      // We can just reuse the existing tree - no need for data streaming

      cl::Event evaluation_complete;
      // Purge any previous results, so that our results are unaffected
      // by the last calclation
      _tree->purge_state();
      _tree->evaluate_tree(_evaluation_points_buffer,
                           _reconstruction_value_sum_state_buffer,
                           _reconstruction_weight_sum_state_buffer,
                           _reconstruction_result,
                           num_evaluation_points,
                           static_cast<device_scalar>(_opening_angle),
                           &evaluation_complete);

      cl_int err = evaluation_complete.wait();
      qcl::check_cl_error(err, "Error while executing tree interpolation kernel");
      this->_num_reconstructed_points = num_evaluation_points;
    }
    else
    {
      reconstruction_quantity::quantity_transformation transformation{
        _ctx, q, _blocksize};

      std::vector<particle> particles;

      particles.reserve(_blocksize);
      std::vector<device_scalar> quantities_of_particle(transformation.get_num_quantities());
      std::vector<device_scalar> transformed_quantities;

      io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();
      io::async_for_each_block(streamer.begin_row_blocks(this->_blocksize),
                               streamer.end_row_blocks(),
                               [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
      {

        if(current_block.get_num_available_rows() == 0)
          // Make sure we actually have work to do...
          return;

        particles.clear();
        transformation.clear();

        for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
        {
          access.select_dataset(0);

          math::vector3 particle_position;
          for(std::size_t j = 0; j < 3; ++j)
            particle_position[j] = access(current_block, i, j);

          coordinate_system::correct_periodicity(_render_volume.get_bounding_box_min_corner(),
                                                 _render_volume.get_bounding_box_max_corner(),
                                                 _render_volume.periodic_wraparound_size,
                                                 0.0,
                                                 particle_position);

          if(_render_volume.contains_point(particle_position))
          {
            // Build prelimary particle
            particle p;
            for(std::size_t j = 0; j < 3; ++j)
              p.s[j] = static_cast<device_scalar>(particle_position[j]);

            particles.push_back(p);

            for(std::size_t j = 0; j < transformation.get_num_quantities(); ++j)
            {
              access.select_dataset(1 + j);
              quantities_of_particle[j] =
                  static_cast<device_scalar>(access(current_block, i));
            }

            transformation.queue_input_quantities(quantities_of_particle);
          }
        }
        std::cout << "Processing "
                  << particles.size()
                  << " particles between "
                  << current_block.get_available_data_range_begin() << " / "
                  << current_block.get_available_data_range_end() << std::endl;

        transformation.commit_data();

        // Transform the quantities
        cl::Event transformation_complete;
        transformation(&transformation_complete);
        cl_int err = transformation_complete.wait();
        qcl::check_cl_error(err, "Error during the quantity transformation.");

        cl::Event transformation_results_transferred;
        transformation.retrieve_results(&transformation_results_transferred,
                                        transformed_quantities);
        err = transformation_results_transferred.wait();
        qcl::check_cl_error(err, "Could not retrieve results from the quantity transformation");

        // Finalize particles
        assert(particles.size() == transformed_quantities.size());
        for(std::size_t i = 0; i < particles.size(); ++i)
          particles[i].s[3] = transformed_quantities[i];

        // Create tree
        _tree = std::make_shared<sparse_interpolation_tree>(particles, _ctx);

        cl::Event evaluation_complete;

        _tree->evaluate_tree(_evaluation_points_buffer,
                             _reconstruction_value_sum_state_buffer,
                             _reconstruction_weight_sum_state_buffer,
                             _reconstruction_result,
                             num_evaluation_points,
                             static_cast<device_scalar>(_opening_angle),
                             &evaluation_complete,
                             true);

        err = evaluation_complete.wait();
        qcl::check_cl_error(err, "Error while executing tree interpolation kernel");

      });
      this->_num_reconstructed_points = num_evaluation_points;
    }
  }

  void run(const std::vector<device_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& q)
  {

    _ctx->create_input_buffer<device_vector4>(_evaluation_points_buffer, evaluation_points.size());

    _ctx->memcpy_h2d(_evaluation_points_buffer,
                     evaluation_points.data(),
                     evaluation_points.size());
    run(_evaluation_points_buffer, evaluation_points.size(), q);
  }

  /// \return The number of reconstructed data points
  std::size_t get_num_reconstructed_points() const
  {
    return _num_reconstructed_points;
  }

  /// \return An OpenCL data buffer containing the result of the reconstruction
  /// if the reconstruction has already been executed. Otherwise, its content
  /// is undefined.
  const cl::Buffer& get_reconstruction() const
  {
    return _reconstruction_result;
  }

  /// \return The OpenCL context
  const qcl::device_context_ptr get_context() const
  {
    return _ctx;
  }

  /// \return The tree used by the reconstruction
  std::shared_ptr<sparse_interpolation_tree> get_tree() const
  {
    return _tree;
  }

private:
  std::size_t _blocksize;

  qcl::device_context_ptr _ctx;
  volume_cutout _render_volume;

  H5::DataSet _coordinates;

  std::shared_ptr<sparse_interpolation_tree> _tree;

  std::size_t _num_reconstructed_points;

  cl::Buffer _evaluation_points_buffer;
  cl::Buffer _reconstruction_result;
  cl::Buffer _reconstruction_value_sum_state_buffer;
  cl::Buffer _reconstruction_weight_sum_state_buffer;

  math::scalar _opening_angle;
};




template<class Volumetric_reconstructor>
class volumetric_slice
{
public:

  volumetric_slice(const camera& cam)
    : _cam{cam}
  {
  }

  void create_slice(Volumetric_reconstructor& reconstruction,
           const reconstruction_quantity::quantity& reconstructed_quantity,
           util::multi_array<device_scalar>& output,
           std::size_t num_additional_samples = 0) const
  {

    output = util::multi_array<device_scalar>{_cam.get_num_pixels(0),
                                              _cam.get_num_pixels(1)};
    std::fill(output.begin(), output.end(), 0.0f);

    std::size_t total_num_pixels = _cam.get_num_pixels(0) *
                                   _cam.get_num_pixels(1);

    std::random_device rd;
    std::mt19937 random(rd());
    std::uniform_real_distribution<math::scalar> uniform(-0.5, 0.5);

    std::size_t samples_per_pixel = num_additional_samples + 1;

    std::vector<device_vector4> evaluation_points(samples_per_pixel * total_num_pixels);

    for(std::size_t i = 0; i < samples_per_pixel; ++i)
    {
      for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
      {
        for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
        {
          if(i == 0)
          {
            math::vector3 pixel_coord = _cam.get_pixel_coordinate(x,y);
            evaluation_points[y * _cam.get_num_pixels(0) + x] =
              vector3_to_cl_vector4(pixel_coord);
          }
          else
          {
            math::scalar sampled_x = static_cast<math::scalar>(x)+uniform(random);
            math::scalar sampled_y = static_cast<math::scalar>(y)+uniform(random);
            math::vector3 coord = _cam.get_pixel_coordinate(sampled_x, sampled_y);

            evaluation_points[y * _cam.get_num_pixels(0) + x + i * total_num_pixels] =
              vector3_to_cl_vector4(coord);
          }
        }
      }
    }

    reconstruction.run(evaluation_points, reconstructed_quantity);

    // Retrieve results
    std::vector<device_scalar> result_buffer(samples_per_pixel * total_num_pixels);
    reconstruction.get_context()->memcpy_d2h(result_buffer.data(),
                                             reconstruction.get_reconstruction(),
                                             samples_per_pixel * total_num_pixels);

#ifdef CPU_TREE_TEST
    for(std::size_t i = 0; i < evaluation_points.size(); ++i)
    {
      sparse_interpolation_tree::scalar weights = 0;
      sparse_interpolation_tree::scalar values = 0;

      math::vector3 point = {{evaluation_points[i].s[0],
                              evaluation_points[i].s[1],
                              evaluation_points[i].s[2]}};
      reconstruction.get_tree()->evaluate_single_point(point, 0.2, values, weights);
      result_buffer[i] = values / weights;
    }
#endif

    math::scalar pixel_volume = _cam.get_pixel_size()
                              * _cam.get_pixel_size()
                              * _cam.get_pixel_size();
    math::scalar dV = reconstructed_quantity.effective_volume_integration_dV(
          pixel_volume * reconstructed_quantity.get_unit_converter().volume_conversion_factor(),
          pixel_volume * reconstructed_quantity.get_unit_converter().volume_conversion_factor());

    for(std::size_t i = 0; i < samples_per_pixel; ++i)
    {
      std::size_t offset = i * total_num_pixels;
      for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
        for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
        {
          std::size_t idx [] = {x,y};
          math::scalar contribution =
              result_buffer[y * _cam.get_num_pixels(0) + x + offset];
          output[idx] += dV * contribution / static_cast<math::scalar>(samples_per_pixel);
        }
    }
  }

private:
  device_vector4 vector3_to_cl_vector4(const math::vector3& x) const
  {
    device_vector4 result;
    for(std::size_t i = 0; i < 3; ++i)
      result.s[i] = static_cast<device_scalar>(x[i]);
    return result;
  }

  camera _cam;
};

template<class Volumetric_reconstructor>
class volumetric_tomography
{
public:

  volumetric_tomography(const camera& cam)
    : _cam{cam}
  {}

  virtual ~volumetric_tomography(){}

  void set_camera(const camera& cam)
  {
    _cam = cam;
  }

  virtual void create_tomographic_cube(Volumetric_reconstructor& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               math::scalar z_range,
                               util::multi_array<device_scalar>& output)
  {

    std::size_t total_num_pixels_z = static_cast<std::size_t>(z_range / _cam.get_pixel_size());
    if(total_num_pixels_z == 0)
      total_num_pixels_z = 1;

    create_tomographic_cube(reconstruction, reconstructed_quantity, 0, total_num_pixels_z, output);

  }

  const camera& get_camera() const
  {
    return _cam;
  }

protected:
  void create_tomographic_cube(Volumetric_reconstructor& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               std::size_t initial_z_step,
                               std::size_t num_steps,
                               util::multi_array<device_scalar>& output)
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
      volumetric_slice<Volumetric_reconstructor> slice{moving_cam};
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


private:

  camera _cam;
};

template<class Volumetric_reconstructor, class Partitioner>
class distributed_volumetric_tomography : public volumetric_tomography<Volumetric_reconstructor>
{
public:
  distributed_volumetric_tomography(const Partitioner& partitioner,
                                    const camera& cam)
    : volumetric_tomography<Volumetric_reconstructor>{cam},
      _partitioner{partitioner}
  {}

  virtual void create_tomographic_cube(Volumetric_reconstructor& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               math::scalar z_range,
                               util::multi_array<device_scalar>& local_result) override
  {

    std::size_t total_num_pixels_z = static_cast<std::size_t>(z_range /
                                                              this->get_camera().get_pixel_size());
    if(total_num_pixels_z == 0)
      total_num_pixels_z = 1;

    _partitioner.run(total_num_pixels_z);

    volumetric_tomography<Volumetric_reconstructor>::create_tomographic_cube(
                                  reconstruction,
                                  reconstructed_quantity,
                                  _partitioner.own_begin(),
                                  _partitioner.own_end() - _partitioner.own_begin(),
                                  local_result);

  }

  virtual ~distributed_volumetric_tomography(){}

  const Partitioner& get_partitioning() const
  {
    return _partitioner;
  }
private:

  Partitioner _partitioner;
};


template<class Volumetric_reconstructor>
class volumetric_integration
{
public:

  using integrator = integration::runge_kutta_fehlberg<device_scalar, math::scalar>;

  volumetric_integration(const qcl::device_context_ptr& ctx,
                         const camera& cam)
    : _cam{cam}, _ctx{ctx}
  {}

  template<class Tolerance_type>
  void create_projection(Volumetric_reconstructor& reconstruction,
                         const reconstruction_quantity::quantity& reconstructed_quantity,
                         math::scalar z_range,
                         const Tolerance_type& integration_tolerance,
                         util::multi_array<device_scalar>& output) const
  {
    output = util::multi_array<device_scalar>{_cam.get_num_pixels(0),
                                              _cam.get_num_pixels(1)};

    std::size_t total_num_pixels = _cam.get_num_pixels(0)
                                 * _cam.get_num_pixels(1);

    std::fill(output.begin(), output.end(), 0.0f);

    integration::parallel_runge_kutta_fehlberg integration_engine{
      _ctx,
      total_num_pixels
    };

    integration::parallel_pixel_integrand<Volumetric_reconstructor> integrand{
      _ctx,
      _cam,
      &reconstruction,
      &reconstructed_quantity
    };

    while(integration_engine.get_num_running_integrators() > 0)
    {
      std::cout << integration_engine.get_num_running_integrators()
                << " integrators are still running.\n";
      integration_engine.advance(integration_tolerance, z_range, integrand);
    }

    // Retrieve results
    _ctx->memcpy_d2h(output.data(),
                     integration_engine.get_integration_state(),
                     total_num_pixels);

    // Finalize results

    device_scalar pixel_area =
        static_cast<device_scalar>(_cam.get_pixel_size() * _cam.get_pixel_size());

    device_scalar length_conversion =
        static_cast<device_scalar>(
          reconstructed_quantity.get_unit_converter().length_conversion_factor());

    device_scalar dA = static_cast<device_scalar>(
          reconstructed_quantity.effective_line_of_sight_integration_dA(
            pixel_area * reconstructed_quantity.get_unit_converter().area_conversion_factor(),
            length_conversion * z_range));

    for(auto it = output.begin(); it != output.end(); ++it)
      (*it) *= length_conversion * dA;
  }


private:
  //static constexpr math::scalar range_epsilon = 0.1;

  camera _cam;
  qcl::device_context_ptr _ctx;
};



}

#endif
