#ifndef VOLUMETRIC_RECONSTRUCTION_HPP
#define VOLUMETRIC_RECONSTRUCTION_HPP

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>

#include "qcl.hpp"
#include "math.hpp"
#include "async_io.hpp"
#include "quantity.hpp"
#include "grid.hpp"
#include "multi_array.hpp"
#include "coordinate_system.hpp"
#include "interpolation_tree.hpp"

#include "integration.hpp"

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

  inline bool contains_point(const math::vector3& point) const
  {
    for(std::size_t i = 0; i < 3; ++i)
    {
      if(point[i] < (center[i] - 0.5 * extent[i]))
        return false;
      if(point[i] > (center[i] + 0.5 * extent[i]))
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

/// Reconstructs the value of a given quantity at a given (arbitrary) set of
/// 3D evaluation points. The values are calculated by a inverse distance interpolation
/// scheme using the 8 nearest neighbors.
class volumetric_nn8_reconstruction
{
public:
  using result_scalar = float;
  using nearest_neighbor_list = cl_float8;
  using result_vector4 = cl_float4;
  using result_vector3 = cl_float3;
  using result_vector2 = cl_float2;

  const std::size_t blocksize = 700000;
  //const std::size_t desired_num_particles_per_tile = 100;
  const math::scalar additional_border_region_size = 100.0;
  const std::size_t local_size1D = 256;
  const std::size_t local_size2D = 16;

  volumetric_nn8_reconstruction(const qcl::device_context_ptr& ctx,
                            const volume_cutout& render_volume,
                            const H5::DataSet& coordinates,
                            const H5::DataSet& smoothing_lengths)
    : _ctx{ctx},
      _render_volume{render_volume},
      _coordinates{coordinates},
      _smoothing_lengths{smoothing_lengths},
      _kernel{ctx->get_kernel("volumetric_reconstruction")},
      _finalization_kernel{ctx->get_kernel("finalize_volumetric_reconstruction")},
      _num_reconstructed_points{0}
  {

  }

  /// Execute the reconstruction
  /// \param evaluation_points The points at which the data shall be
  /// interpolated
  /// \param quantity_to_reconstruct The quantity that shall be estimated
  void run(const std::vector<result_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct)
  {
    std::vector<H5::DataSet> streamed_quantities{
      _coordinates,
      _smoothing_lengths
    };

    for(const H5::DataSet& quantity : quantity_to_reconstruct.get_required_datasets())
      streamed_quantities.push_back(quantity);

    io::async_dataset_streamer<math::scalar> streamer{streamed_quantities};

    //math::scalar mean_particle_density = blocksize / _render_volume.get_volume();
    math::scalar tile_diameter = 25.0;//std::cbrt(desired_num_particles_per_tile / mean_particle_density);
    auto tiles_extent = _render_volume.get_extent(additional_border_region_size);

    std::array<long long int, 3> num_tiles{
      {static_cast<long long int>(tiles_extent[0] / tile_diameter),
       static_cast<long long int>(tiles_extent[1] / tile_diameter),
       static_cast<long long int>(tiles_extent[2] / tile_diameter)
      }
    };

    for(std::size_t i = 0; i < 3; ++i)
      if(num_tiles[i] < 1)
        num_tiles[i] = 1;

    util::grid_coordinate_translator<3> tile_grid{
      _render_volume.center,
      tiles_extent,
      num_tiles
    };

    util::multi_array<result_vector4> tiles{
      static_cast<std::size_t>(num_tiles[0]),
      static_cast<std::size_t>(num_tiles[1]),
      static_cast<std::size_t>(num_tiles[2])
    };

    std::cout << "Using "
              << num_tiles[0] << "/"
              << num_tiles[1] << "/"
              << num_tiles[2] << " tiles." << std::endl;

    std::vector<result_vector4> particles(blocksize);
    std::vector<result_scalar> smoothing_distances(blocksize);
    std::vector<result_vector4> filtered_particles(blocksize);
    std::vector<std::vector<result_scalar>> input_quantities(
          quantity_to_reconstruct.get_required_datasets().size());

    cl::Buffer quantity_buffer;
    std::vector<cl::Buffer> input_quantities_buffer(
          quantity_to_reconstruct.get_required_datasets().size());
    cl::Buffer tiles_buffer;
    cl::Buffer particles_buffer;
    cl::Buffer evaluation_point_coordinate_buffer;
    cl::Buffer evaluation_point_weights_buffer;
    cl::Buffer evaluation_point_values_buffer;

    _ctx->create_buffer<result_scalar>(quantity_buffer, CL_MEM_READ_WRITE, blocksize);
    for(std::size_t i = 0; i < input_quantities_buffer.size(); ++i)
    {
      _ctx->create_input_buffer<result_scalar>(input_quantities_buffer[i], blocksize);
      input_quantities[i] = std::vector<result_scalar>(blocksize);
    }
    _ctx->create_input_buffer<result_vector4>(tiles_buffer, tiles.get_num_elements());
    _ctx->create_input_buffer<result_vector4>(particles_buffer, blocksize);
    _ctx->create_input_buffer<result_vector4>(evaluation_point_coordinate_buffer, evaluation_points.size());
    _ctx->create_buffer<nearest_neighbor_list>(evaluation_point_weights_buffer,
                                               CL_MEM_READ_WRITE,
                                               evaluation_points.size());
    _ctx->create_buffer<nearest_neighbor_list>(evaluation_point_values_buffer,
                                               CL_MEM_READ_WRITE,
                                               evaluation_points.size());

    // Copy evaluation points to the GPU
    _ctx->memcpy_h2d(evaluation_point_coordinate_buffer,
                     evaluation_points.data(),
                     evaluation_points.size());

    bool is_first_block = true;

    io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();

    cl::Event kernel_finished;
    cl_int err;

    io::async_for_each_block(streamer.begin_row_blocks(blocksize),
                             streamer.end_row_blocks(),
                             [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
    {
      // Don't bother if there's nothing to do..
      if(current_block.get_num_available_rows() == 0)
        return;

      // Filter the particles - only include those that are within
      // the render volume
      math::scalar maximum_smoothing_distance = 0.0;

      std::size_t num_filtered_particles = filter_particles(current_block,
                                                            access,
                                                            tile_grid,
                                                            filtered_particles,
                                                            quantity_to_reconstruct,
                                                            input_quantities,
                                                            maximum_smoothing_distance,
                                                            smoothing_distances);

      std::cout << "Processing "
                << num_filtered_particles
                << " particles between "
                << current_block.get_available_data_range_begin() << " / "
                << current_block.get_available_data_range_end() << std::endl;

      // Make sure we actually have work to do for
      // this block before involving the GPU
      if(num_filtered_particles > 0)
      {
        // Start transforming the quantities
        cl::Event quantity_transformed_event;
        transform_quantity(input_quantities,
                           quantity_to_reconstruct,
                           input_quantities_buffer,
                           quantity_buffer,
                           num_filtered_particles,
                           &quantity_transformed_event);

        // Fill tiles and sort particles into tiles
        sort_into_tiles(filtered_particles,
                      smoothing_distances,
                      num_filtered_particles,
                      tile_grid,
                      tiles,
                      particles);
        // Wait for the previous kernel to finish
        if(!is_first_block)
        {
          err = kernel_finished.wait();
          qcl::check_cl_error(err, "Error while waiting for the volumetric reconstruction kernel to finish.");
        }


        // Start moving data to the GPU
        cl::Event tiles_transferred_event, particles_transferred_event;
        _ctx->memcpy_h2d_async(tiles_buffer,
                               tiles.data(),
                               tiles.get_num_elements(),
                               &tiles_transferred_event);

        _ctx->memcpy_h2d_async(particles_buffer,
                               particles.data(),
                               blocksize,
                               &particles_transferred_event);


        err = quantity_transformed_event.wait();
        qcl::check_cl_error(err, "Error while waiting for the quantity transformation to complete");

        // Launch kernel
        std::vector<cl::Event> dependencies = {tiles_transferred_event,
                                               particles_transferred_event};


        math::vector3 tiles_min_corner = tile_grid.get_grid_min_corner();
        math::vector3 tile_sizes = tile_grid.get_cell_sizes();

        qcl::kernel_argument_list arguments{_kernel};
        arguments.push(static_cast<cl_int>(is_first_block));
        arguments.push(tiles_buffer);
        arguments.push(cl_int3{{static_cast<cl_int>(num_tiles[0]),
                                static_cast<cl_int>(num_tiles[1]),
                                static_cast<cl_int>(num_tiles[2])}});

        arguments.push(result_vector3{{static_cast<result_scalar>(tiles_min_corner[0]),
                                       static_cast<result_scalar>(tiles_min_corner[1]),
                                       static_cast<result_scalar>(tiles_min_corner[2])}});

        arguments.push(result_vector3{{static_cast<result_scalar>(tile_sizes[0]),
                                       static_cast<result_scalar>(tile_sizes[1]),
                                       static_cast<result_scalar>(tile_sizes[2])}});

        arguments.push(particles_buffer);
        arguments.push(static_cast<result_scalar>(maximum_smoothing_distance));
        arguments.push(static_cast<cl_int>(evaluation_points.size()));
        arguments.push(evaluation_point_coordinate_buffer);
        arguments.push(evaluation_point_weights_buffer);
        arguments.push(evaluation_point_values_buffer);
        arguments.push(quantity_buffer);



        err = _ctx->get_command_queue().enqueueNDRangeKernel(*_kernel,
                                                             cl::NullRange,
                                                             cl::NDRange(math::make_multiple_of(local_size1D,
                                                                                        evaluation_points.size())),
                                                             cl::NDRange(local_size1D), &dependencies, &kernel_finished);
        qcl::check_cl_error(err, "Could not enqueue volumetric reconstruction kernel");


        is_first_block = false;
      }
    });

    // Wait for the last kernel to finish
    if(!is_first_block)
    {
      err = kernel_finished.wait();
      qcl::check_cl_error(err, "Error while waiting for the volumetric reconstruction kernel to finish.");
    }


    _num_reconstructed_points = evaluation_points.size();
    _ctx->create_buffer<result_scalar>(_reconstruction_result,
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
                                                                                      evaluation_points.size())),
                                                   cl::NDRange(local_size1D), nullptr, &finalization_done);
    qcl::check_cl_error(err, "Could not enqueue finalization kernel.");

    err = finalization_done.wait();
    qcl::check_cl_error(err, "Error while waiting for finalization kernel to finish.");
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

  /// Filters the particles in the current block such that only particles
  /// within the rendering volume remain, discarding any other particles.
  /// Additionally, the particles are transformed into the float4 vector
  /// format used by the GPU:
  /// particle.x -- x coordinate
  /// particle.y -- y coordinate
  /// particle.z -- z coordinate
  /// particle.w -- particle id i, such that in the quantity buffer
  /// quantities[i] refers to the value associated with this particle.
  /// Also estimates the maximum smoothing distance of the particle
  /// collection, and fills the quantities buffer.
  std::size_t filter_particles(const io::async_dataset_streamer<math::scalar>::const_iterator& current_block,
                        io::buffer_accessor<math::scalar>& access,
                        const util::grid_coordinate_translator<3>& tile_grid,
                        std::vector<result_vector4>& filtered_particles,
                        const reconstruction_quantity::quantity& quantity_to_reconstruct,
                        std::vector<std::vector<result_scalar>>& quantities,
                        math::scalar& max_smoothing_distance,
                        std::vector<result_scalar>& smoothing_distances) const
  {

    assert(filtered_particles.size() >= current_block.get_num_available_rows());
    for(std::size_t i = 0; i < quantities.size(); ++i)
      assert(quantities[i].size() >= current_block.get_num_available_rows());
    assert(smoothing_distances.size() >= current_block.get_num_available_rows());

    std::size_t num_filtered_particles = 0;
    max_smoothing_distance = 0.0;
    auto scaling_factors = quantity_to_reconstruct.get_quantitiy_scaling_factors();

    for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
    {
      access.select_dataset(0);
      math::vector3 coordinate;
      for(std::size_t j = 0; j < 3; ++j)
        // The static_cast ensures that the rounding errors
        // remain consistent
        coordinate[j] = static_cast<result_scalar>(access(current_block, i, j));

      access.select_dataset(1);
      math::scalar smoothing_length = 0.5 * access(current_block, i);

      if(smoothing_length > max_smoothing_distance)
        max_smoothing_distance = smoothing_length;


      coordinate_system::correct_periodicity(tile_grid,
                                             _render_volume.periodic_wraparound_size,
                                             smoothing_length,
                                             coordinate);
      auto grid_idx = tile_grid(coordinate);
      // Filter the particles by position - only include those which
      // are within the tile grid

      if(tile_grid.is_within_bounds(grid_idx))
      {
        // Build particle
        result_vector4 particle;
        for(std::size_t j = 0; j < 3; ++j)
          particle.s[j] = static_cast<result_scalar>(coordinate[j]);
        particle.s[3] = static_cast<result_scalar>(num_filtered_particles);

        filtered_particles[num_filtered_particles] = particle;

        // Retrieve quantities belonging to the particle
        for(std::size_t j = 0; j < quantities.size(); ++j)
        {
          access.select_dataset(2 + j);

          quantities[j][num_filtered_particles] =
              static_cast<result_scalar>(scaling_factors[j] * access(current_block, i));
        }
        smoothing_distances[num_filtered_particles] =
            static_cast<result_scalar>(smoothing_length);

        ++num_filtered_particles;
      }
    }
    return num_filtered_particles;
  }

  /// Applies the quantity transformation Q based on input quantities q_i
  /// such that output_j = Q(q_0j, q_1j, ..., q_ij). This calculation
  /// is done asynchronously on the GPU, i.e. it is mandatory to wait for
  /// the Event \c evt before accessing the \c output_buffer.
  void transform_quantity(const std::vector<std::vector<result_scalar>>& input_quantities,
                          const reconstruction_quantity::quantity& quantity_to_reconstruct,
                          const std::vector<cl::Buffer>& input_buffers,
                          const cl::Buffer& output_buffer,
                          std::size_t num_filtered_particles,
                          cl::Event* evt) const
  {
    assert(input_quantities.size() == input_buffers.size());

    for(std::size_t j = 0; j < input_quantities.size(); ++j)
      _ctx->memcpy_h2d<result_scalar>(input_buffers[j], input_quantities[j].data(),
                                      num_filtered_particles);

    qcl::kernel_ptr kernel = quantity_to_reconstruct.get_kernel(_ctx);

    qcl::kernel_argument_list arguments{kernel};
    arguments.push(output_buffer);
    arguments.push(static_cast<cl_uint>(num_filtered_particles));
    for(std::size_t j = 0; j < input_buffers.size(); ++j)
      arguments.push(input_buffers[j]);


    cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(*kernel,
                                                   cl::NullRange,
                                                   cl::NDRange(math::make_multiple_of(local_size1D,
                                                                                      num_filtered_particles)),
                                                   cl::NDRange(local_size1D),
                                                   nullptr,
                                                   evt);
    qcl::check_cl_error(err, "Could not enqueue quantity transformation kernel");
  }

  /// Prepares the tile data structure and sorts particles into their tiles
  void sort_into_tiles(const std::vector<result_vector4>& filtered_particles,
                     const std::vector<result_scalar>& smoothing_distances,
                     std::size_t num_filtered_particles,
                     const util::grid_coordinate_translator<3>& tile_grid,
                     util::multi_array<result_vector4>& tiles_buffer,
                     std::vector<result_vector4>& particles)
  {

    std::fill(tiles_buffer.begin(),
              tiles_buffer.end(),
              result_vector4{{0.0f, 0.0f, 0.0f, 0.0f}});

    // Count the number of particles for each tile
    for(std::size_t i = 0; i < num_filtered_particles; ++i)
    {
      result_vector4 particle = filtered_particles[i];

      math::vector3 particle_coordinates;
      for(std::size_t j = 0; j < 3; ++j)
        particle_coordinates[j] = static_cast<math::scalar>(particle.s[j]);

      auto grid_idx = tile_grid(particle_coordinates);
      assert(tile_grid.is_within_bounds(grid_idx));

      auto grid_uidx = tile_grid.unsigned_grid_index(grid_idx);
      tiles_buffer[grid_uidx].s[0] += 1.0f;
    }

    // Set offsets for particles
    for(std::size_t i = 0; i < tiles_buffer.get_num_elements(); ++i)
    {
      result_scalar previous_offset = 0.0;
      result_scalar previous_num_particles = 0.0;
      if(i > 0)
      {
        previous_offset        = tiles_buffer.data()[i-1].s[2];
        previous_num_particles = tiles_buffer.data()[i-1].s[0];
      }
      tiles_buffer.data()[i].s[2] = previous_offset + previous_num_particles;
    }

    // Sort particles
    for(std::size_t i = 0; i < num_filtered_particles; ++i)
    {
      result_vector4 particle = filtered_particles[i];

      math::vector3 particle_coordinates;
      for(std::size_t j = 0; j < 3; ++j)
        particle_coordinates[j] = static_cast<math::scalar>(particle.s[j]);

      auto grid_idx = tile_grid(particle_coordinates);

      assert(tile_grid.is_within_bounds(grid_idx));

      auto grid_uidx = tile_grid.unsigned_grid_index(grid_idx);

      std::size_t already_placed_particles =
          static_cast<std::size_t>(tiles_buffer[grid_uidx].s[1]);

      std::size_t offset =
          static_cast<std::size_t>(tiles_buffer[grid_uidx].s[2]);

      particles[already_placed_particles + offset] = particle;

      // Increase number of already placed particles
      tiles_buffer[grid_uidx].s[1] += 1;
      // Update maximum smoothing length of tile
      if(tiles_buffer[grid_uidx].s[3] < smoothing_distances[i])
        tiles_buffer[grid_uidx].s[3] = smoothing_distances[i];
    }


  }

  qcl::device_context_ptr _ctx;
  volume_cutout _render_volume;

  H5::DataSet _coordinates;
  H5::DataSet _smoothing_lengths;

  qcl::kernel_ptr _kernel;
  qcl::kernel_ptr _finalization_kernel;

  std::size_t _num_reconstructed_points;
  cl::Buffer  _reconstruction_result;

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

  using result_scalar = float;
  using nearest_neighbor_list = cl_float8;
  using result_vector4 = cl_float4;
  using result_vector3 = cl_float3;
  using result_vector2 = cl_float2;
  using particle = cl_float4;


  void run(const std::vector<result_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& q)
  {
    std::vector<H5::DataSet> streamed_data = {{_coordinates}};
    auto required_datasets = q.get_required_datasets();
    for(const H5::DataSet& data : required_datasets)
      streamed_data.push_back(data);

    io::async_dataset_streamer<math::scalar> streamer{streamed_data};

    _ctx->create_input_buffer<result_vector4>(_evaluation_points_buffer, evaluation_points.size());
    _ctx->create_buffer<result_scalar>(_reconstruction_result,
                                       CL_MEM_READ_WRITE,
                                       evaluation_points.size());
    _ctx->create_buffer<result_scalar>(_reconstruction_value_sum_state_buffer,
                                       CL_MEM_READ_WRITE,
                                       evaluation_points.size());
    _ctx->create_buffer<result_scalar>(_reconstruction_weight_sum_state_buffer,
                                       CL_MEM_READ_WRITE,
                                       evaluation_points.size());

    cl::Event evaluation_points_transferred;
    _ctx->memcpy_h2d_async(_evaluation_points_buffer,
                           evaluation_points.data(),
                           evaluation_points.size(),
                           &evaluation_points_transferred);

    if((_tree != nullptr) &&
       (streamer.get_num_rows() <= _blocksize))
    {
      // We can just reuse the existing tree - no need for data streaming
      cl_int err = evaluation_points_transferred.wait();
      qcl::check_cl_error(err, "Error while waiting for the evaluation points to be transferred.");

      cl::Event evaluation_complete;
      _tree->evaluate_tree(_evaluation_points_buffer,
                           _reconstruction_value_sum_state_buffer,
                           _reconstruction_weight_sum_state_buffer,
                           _reconstruction_result,
                           evaluation_points.size(),
                           static_cast<result_scalar>(_opening_angle),
                           &evaluation_complete);

      err = evaluation_complete.wait();
      qcl::check_cl_error(err, "Error while executing tree interpolation kernel");
      this->_num_reconstructed_points = evaluation_points.size();
    }
    else
    {
      reconstruction_quantity::quantity_transformation transformation{
        _ctx, q, _blocksize};

      std::vector<particle> particles;

      particles.reserve(_blocksize);
      std::vector<result_scalar> quantities_of_particle(transformation.get_num_quantities());
      std::vector<result_scalar> transformed_quantities;

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
              p.s[j] = static_cast<result_scalar>(particle_position[j]);

            particles.push_back(p);

            for(std::size_t j = 0; j < transformation.get_num_quantities(); ++j)
            {
              access.select_dataset(1 + j);
              quantities_of_particle[j] =
                  static_cast<result_scalar>(access(current_block, i));
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


        err = evaluation_points_transferred.wait();
        qcl::check_cl_error(err, "Error while waiting for the evaluation points to be transferred.");

        cl::Event evaluation_complete;

        _tree->evaluate_tree(_evaluation_points_buffer,
                             _reconstruction_value_sum_state_buffer,
                             _reconstruction_weight_sum_state_buffer,
                             _reconstruction_result,
                             evaluation_points.size(),
                             static_cast<result_scalar>(_opening_angle),
                             &evaluation_complete);

        err = evaluation_complete.wait();
        qcl::check_cl_error(err, "Error while executing tree interpolation kernel");

      });
      this->_num_reconstructed_points = evaluation_points.size();
    }
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

class camera
{
public:
  camera(const math::vector3& position,
         const math::vector3& look_at,
         math::scalar roll_angle,
         math::scalar screen_width,
         std::size_t num_pix_x,
         std::size_t num_pix_y)
    : _position{position},
      _look_at{look_at},
      _pixel_size{screen_width/static_cast<math::scalar>(num_pix_x)},
      _num_pixels{{num_pix_x, num_pix_y}}
  {
    // Calculate screen basis vectors
    math::vector3 v1 {{0, 0, 1}};

    if (_look_at[0] == v1[0] &&
        _look_at[1] == v1[1] &&
        _look_at[2] == v1[2])
    {
      v1 = {{0, 1, 0}};
    }

    v1 = math::cross(v1, _look_at);
    math::vector3 v2 = math::cross(_look_at, v1);

    math::matrix3x3 roll_matrix;
    math::matrix_create_rotation_matrix(&roll_matrix, _look_at, roll_angle);

    // Normalize vectors in case of rounding errors
    this->_screen_basis_vector0 =
        math::normalize(math::matrix_vector_mult(roll_matrix, v1));

    this->_screen_basis_vector1 =
        math::normalize(math::matrix_vector_mult(roll_matrix, v2));

    update_min_position();
  }

  std::size_t get_num_pixels(std::size_t dim) const
  {
    return _num_pixels[dim];
  }

  math::vector3 get_pixel_coordinate(std::size_t x_index, std::size_t y_index) const
  {
    return _min_position
         + x_index * _pixel_size * _screen_basis_vector0
         + y_index * _pixel_size * _screen_basis_vector1;
  }

  math::vector3 get_pixel_coordinate(math::scalar x_index, math::scalar y_index) const
  {
    return _min_position
         + x_index * _pixel_size * _screen_basis_vector0
         + y_index * _pixel_size * _screen_basis_vector1;
  }

  math::scalar get_pixel_size() const
  {
    return _pixel_size;
  }

  const math::vector3& get_position() const
  {
    return _position;
  }

  const math::vector3& get_look_at() const
  {
    return _look_at;
  }

  void set_position(const math::vector3& pos)
  {
    this->_position = pos;
    update_min_position();
  }

private:
  void update_min_position()
  {
    _min_position = _position;
    _min_position -= (_num_pixels[0]/2.0) * _pixel_size * _screen_basis_vector0;
    _min_position -= (_num_pixels[1]/2.0) * _pixel_size * _screen_basis_vector1;
  }

  math::vector3 _position;
  math::vector3 _look_at;

  math::vector3 _screen_basis_vector0;
  math::vector3 _screen_basis_vector1;

  math::scalar _pixel_size;

  std::array<std::size_t, 2> _num_pixels;

  math::vector3 _min_position;
};



template<class Volumetric_reconstructor>
class volumetric_slice
{
public:
  using result_scalar = volumetric_nn8_reconstruction::result_scalar;
  using result_vector4 = volumetric_nn8_reconstruction::result_vector4;

  volumetric_slice(const camera& cam)
    : _cam{cam}
  {
  }

  void create_slice(Volumetric_reconstructor& reconstruction,
           const reconstruction_quantity::quantity& reconstructed_quantity,
           util::multi_array<result_scalar>& output,
           std::size_t num_additional_samples = 0) const
  {

    output = util::multi_array<result_scalar>{_cam.get_num_pixels(0),
                                              _cam.get_num_pixels(1)};
    std::fill(output.begin(), output.end(), 0.0f);

    std::size_t total_num_pixels = _cam.get_num_pixels(0) *
                                   _cam.get_num_pixels(1);

    std::random_device rd;
    std::mt19937 random(rd());
    std::uniform_real_distribution<math::scalar> uniform(-0.5, 0.5);

    std::size_t samples_per_pixel = num_additional_samples + 1;

    std::vector<result_vector4> evaluation_points(samples_per_pixel * total_num_pixels);

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
    std::vector<result_scalar> result_buffer(samples_per_pixel * total_num_pixels);
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
    math::scalar dV = 1.0;
    if(reconstructed_quantity.is_integrated_quantity())
      dV = _cam.get_pixel_size()
         * _cam.get_pixel_size()
         * _cam.get_pixel_size();

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
  result_vector4 vector3_to_cl_vector4(const math::vector3& x) const
  {
    result_vector4 result;
    for(std::size_t i = 0; i < 3; ++i)
      result.s[i] = static_cast<result_scalar>(x[i]);
    return result;
  }

  camera _cam;
};

template<class Volumetric_reconstructor>
class volumetric_tomography
{
public:
  using result_scalar = volumetric_nn8_reconstruction::result_scalar;

  volumetric_tomography(const camera& cam)
    : _cam{cam}
  {}

  void create_tomographic_cube(Volumetric_reconstructor& reconstruction,
                               const reconstruction_quantity::quantity& reconstructed_quantity,
                               math::scalar z_range,
                               util::multi_array<result_scalar>& output)
  {
    std::size_t num_pixels_z = static_cast<std::size_t>(z_range / _cam.get_pixel_size());
    if(num_pixels_z == 0)
      num_pixels_z = 1;

    output = util::multi_array<result_scalar>{_cam.get_num_pixels(0),
                                              _cam.get_num_pixels(1),
                                              num_pixels_z};

    std::fill(output.begin(), output.end(), 0.0f);

    camera moving_cam = _cam;

    for(std::size_t z = 0; z < num_pixels_z; ++z)
    {
      std::cout << "z = " << z << std::endl;

      moving_cam.set_position(_cam.get_position()
                              + z * _cam.get_pixel_size() * moving_cam.get_look_at());

      util::multi_array<result_scalar> slice_data;
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


template<class Volumetric_reconstructor>
class volumetric_integration
{
public:
  using result_scalar = typename Volumetric_reconstructor::result_scalar;
  using result_vector4 = typename Volumetric_reconstructor::result_vector4;
  using integrator = integration::runge_kutta_fehlberg<result_scalar, math::scalar>;

  volumetric_integration(const camera& cam)
    : _cam{cam}
  {}

  void create_projection(Volumetric_reconstructor& reconstruction,
                         const reconstruction_quantity::quantity& reconstructed_quantity,
                         math::scalar z_range,
                         math::scalar integration_tolerance,
                         util::multi_array<result_scalar>& output)
  {
    output = util::multi_array<result_scalar>{_cam.get_num_pixels(0),
                                              _cam.get_num_pixels(1)};

    std::fill(output.begin(), output.end(), 0.0f);
    std::size_t total_num_pixels = _cam.get_num_pixels(0)
                                 * _cam.get_num_pixels(1);

    std::vector<integrator> integrators(total_num_pixels);
    std::vector<std::size_t> integrator_ids(total_num_pixels);

    std::vector<result_scalar> integrand_values(
          integrator::required_num_evaluations * total_num_pixels);

    std::vector<result_vector4> evaluation_points;
    evaluation_points.reserve(integrator::required_num_evaluations * total_num_pixels);


    std::size_t num_running_integrators = 0;
    do
    {
      evaluation_points.clear();

      num_running_integrators = 0;
      for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
      {
        for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
        {
          std::size_t pos = y * _cam.get_num_pixels(0) + x;
          if(integrators[pos].get_position() < z_range)
          {
            math::vector3 pixel_coord = _cam.get_pixel_coordinate(x,y);


            typename integrator::evaluation_coordinates required_evaluations;
            integrators[pos].obtain_next_step_coordinates(required_evaluations);

            for(std::size_t i = 0; i < integrator::required_num_evaluations; ++i)
            {
              math::vector3 coord = pixel_coord;
              coord += _cam.get_look_at() * required_evaluations[i];
              result_vector4 evaluation_point;
              for(std::size_t j = 0; j < 3; ++j)
                evaluation_point.s[j] = static_cast<result_scalar>(coord[j]);

              evaluation_points.push_back(evaluation_point);
            }
            integrator_ids[num_running_integrators] = pos;
            ++num_running_integrators;
          }
        }
      }

      reconstruction.run(evaluation_points, reconstructed_quantity);
      // Retrieve result
      reconstruction.get_context()->memcpy_d2h(integrand_values.data(),
                                              reconstruction.get_reconstruction(),
                                              evaluation_points.size());
      // Advance integrators
      for(std::size_t i = 0; i < num_running_integrators; ++i)
      {
        std::size_t integrator_id = integrator_ids[i];
        typename integrator::integrand_values values;

        for(std::size_t j = 0; j < integrator::required_num_evaluations; ++j)
          values[j] = integrand_values[integrator::required_num_evaluations * i + j];

        integrators[integrator_id].advance(values, integration_tolerance, z_range);
      }
      std::cout << num_running_integrators << " integrators are still running.\n";
    }
    while(num_running_integrators > 0);

    // Store result
    for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
      for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
      {
        std::size_t pos = y * _cam.get_num_pixels(0) + x;
        std::size_t idx [] = {x,y};

        output[idx] = integrators[pos].get_state();
      }
  }

private:
  //static constexpr math::scalar range_epsilon = 0.1;

  camera _cam;
};

}

#endif
