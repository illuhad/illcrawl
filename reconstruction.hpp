#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <functional>

#include "async_io.hpp"
#include "multi_array.hpp"
#include "grid.hpp"
#include "math.hpp"
#include "quantity.hpp"
#include "coordinate_system.hpp"

#include "qcl_module.hpp"
#include "qcl.hpp"

namespace illcrawl {


//template<class Input_data_type>
class smoothed_quantity_reconstruction2D
{
public:
  using result_scalar = float;
  using result_vector4 = cl_float4;
  using result_vector3 = cl_float3;
  using result_vector2 = cl_float2;

  const std::size_t blocksize = 1000000;

  using coordinate_transformator = std::function<math::vector3(const math::vector3)>;

  smoothed_quantity_reconstruction2D(const qcl::device_context_ptr& ctx,
                                     const std::string& reconstruction_kernel2d_name
                                     ="image_tile_based_reconstruction2D")
    : _ctx{ctx},
      _reconstruction_kernel{ctx->get_kernel(reconstruction_kernel2d_name)},
      _coordinate_transformation{[](const math::vector3& v){return v;}}
  {}

  void set_coordinate_transformation(coordinate_transformator transformator)
  {
    _coordinate_transformation = transformator;
  }

  void run(const reconstruction_quantity::quantity& quantity_to_reconstruct,
                              const H5::DataSet& coordinates,
                              const H5::DataSet& smoothing_lengths,
                              const H5::DataSet& volumes,
                              const math::vector3& center,
                              math::scalar x_size,
                              math::scalar y_size,
                              std::size_t num_pix_x,
                              const math::vector3& periodic_wraparound_size,
                              util::multi_array<result_scalar>& out)
  {
    this->run(quantity_to_reconstruct.get_required_datasets(),
                                 quantity_to_reconstruct.get_quantitiy_scaling_factors(),
                                 coordinates,
                                 smoothing_lengths,
                                 volumes,
                                 center,
                                 x_size,
                                 y_size,
                                 num_pix_x,
                                 periodic_wraparound_size,
                                 quantity_to_reconstruct.get_kernel(_ctx),
                                 out);
  }

  void run(const std::vector<H5::DataSet>& reconstruction_quantities,
           const std::vector<math::scalar> quantities_scaling,
           const H5::DataSet& coordinates, const H5::DataSet& smoothing_lengths,
           const H5::DataSet& volumes, const math::vector3& center,
           math::scalar x_size, math::scalar y_size, std::size_t num_pix_x,
           const math::vector3& periodic_wraparound_size,
           const qcl::kernel_ptr& quantity_transformation_kernel,
           util::multi_array<result_scalar>& out)
  {
    assert(reconstruction_quantities.size() != 0);
    assert(quantities_scaling.size() == reconstruction_quantities.size());

    _ctx->require_several_command_queues(2);
    // Setup grid
    util::grid_coordinate_translator<2> pixel_grid_translator{{{center[0], center[1]}},
                                                        {{x_size, y_size}}, num_pix_x};

    auto grid_min_coordinates = pixel_grid_translator.get_grid_min_corner();
    auto grid_max_coordinates = pixel_grid_translator.get_grid_max_corner();
    result_vector2 cl_grid_min_corner, cl_grid_max_corner;
    for(std::size_t i = 0; i < 2; ++i)
    {
      cl_grid_min_corner.s[i] = static_cast<result_scalar>(grid_min_coordinates[i]);
      cl_grid_max_corner.s[i] = static_cast<result_scalar>(grid_max_coordinates[i]);
    }

    long long num_tiles_x =
        math::make_multiple_of(_local_group_size, pixel_grid_translator.get_num_cells()[0])
        / _local_group_size;
    long long num_tiles_y =
        math::make_multiple_of(_local_group_size, pixel_grid_translator.get_num_cells()[1])
        / _local_group_size;

    util::grid_coordinate_translator<2> tiles_grid_translator{{{center[0], center[1]}},
                                                              {{x_size, y_size}},
                                                              {{num_tiles_x, num_tiles_y}}};

    // Setup data streaming
    std::vector<H5::DataSet> input_data = {coordinates, volumes, smoothing_lengths};
    for(const auto& dataset : reconstruction_quantities)
      input_data.push_back(dataset);

    io::async_dataset_streamer<math::scalar> streamer{input_data};

    // Setup the OpenCL (input data) buffers
    // First, the host side buffers
    std::vector<std::vector<result_scalar>> cl_quantities(reconstruction_quantities.size());
    for(std::size_t i = 0; i < cl_quantities.size(); ++i)
      cl_quantities[i].resize(blocksize);

    std::vector<result_vector4> cl_tile_buffer(num_tiles_x * num_tiles_y);
    std::vector<result_vector4> cl_particles(blocksize);

    // Then the device side
    std::vector<cl::Buffer> quantities_buffer(reconstruction_quantities.size());
    cl::Buffer transformed_quantity_buffer;

    cl::Buffer tiles_buffer;
    _ctx->create_input_buffer<result_vector4>(tiles_buffer, num_tiles_x*num_tiles_y);
    cl::Buffer particles_buffer;
    _ctx->create_input_buffer<result_vector4>(particles_buffer, blocksize);

    for(std::size_t i = 0; i < quantities_buffer.size(); ++i)
      _ctx->create_input_buffer<result_scalar>(quantities_buffer[i], blocksize);

    _ctx->create_buffer<result_scalar>(transformed_quantity_buffer,
                                       CL_MEM_READ_WRITE,
                                       blocksize);


    // Setup output
    out = util::multi_array<result_scalar>{pixel_grid_translator.get_num_cells()[0],
                                           pixel_grid_translator.get_num_cells()[1]};
    std::fill(out.begin(), out.end(), 0.0);

    cl::Buffer output_buffer;
    _ctx->create_buffer<result_scalar>(output_buffer,
                                       CL_MEM_READ_WRITE,
                                       out.get_extent_of_dimension(0) * out.get_extent_of_dimension(1),
                                              out.data());


    // Buffer to count the number of particles in each tile
    std::vector<cl_uint> num_particles_in_tile(num_tiles_x * num_tiles_y, 0);
    std::vector<result_vector4> particles;
    particles.reserve(blocksize);

    io::buffer_accessor<math::scalar> access = streamer.create_buffer_accessor();
    auto block_processor =
        [&](const io::async_dataset_streamer<math::scalar>::const_iterator& current_block)
    {
      std::cout << "Processing block with elements "
                << current_block.get_available_data_range_begin()
                << " to "
                << current_block.get_available_data_range_end() << std::endl;

      std::fill(num_particles_in_tile.begin(), num_particles_in_tile.end(), 0);
      particles.clear();

      math::scalar max_smoothing_length = 0.0;
      for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
      {
        access.select_dataset(0);
        math::vector3 particle_position;
        for(std::size_t j = 0; j < 3; ++j)
          particle_position[j] = access(current_block, i, j);

        access.select_dataset(1);
        math::scalar volume = access(current_block, i);

        access.select_dataset(2);
        math::scalar original_smoothing_length = access(current_block, i);

        math::scalar smoothing_length =
            std::min(_smoothing_length_scale_factor * std::cbrt(3./(4.*M_PI) * volume),
                     0.5 * original_smoothing_length);


        if(smoothing_length > max_smoothing_length)
          max_smoothing_length = smoothing_length;

        // Correct for periodicity of the simulation volume
        coordinate_system::correct_periodicity(pixel_grid_translator,
                                               periodic_wraparound_size,
                                               smoothing_length,
                                               particle_position);

        for(std::size_t j = 0; j < reconstruction_quantities.size(); ++j)
        {
          access.select_dataset(3 + j);
          math::scalar q = access(current_block, i);

          cl_quantities[j][i] = static_cast<result_scalar>(quantities_scaling[j] * q);
        }

        // Apply coordinate transformations
        particle_position = _coordinate_transformation(particle_position);

        result_vector4 particle;
        for(std::size_t j = 0; j < 2; ++j)
          particle.s[j] = static_cast<result_scalar>(particle_position[j]);
        particle.s[2] = static_cast<result_scalar>(smoothing_length);
        particle.s[3] = static_cast<result_scalar>(i);

        // Filter particles
        if(grid_min_coordinates[0] <= (particle.s[0] + _additional_border_region_size) &&
           grid_min_coordinates[1] <= (particle.s[1] + _additional_border_region_size) &&
           grid_max_coordinates[0] >= (particle.s[0] - _additional_border_region_size) &&
           grid_max_coordinates[1] >= (particle.s[1] - _additional_border_region_size))
        {
          particles.push_back(particle);

          auto tile = tiles_grid_translator({{particle_position[0], particle_position[1]}});
          tiles_grid_translator.clamp_grid_index_to_edge(tile);
          num_particles_in_tile[static_cast<std::size_t>(tile[1] * num_tiles_x + tile[0])] += 1;
        }
      }


      /*
      for(std::size_t y = 0; y < num_tiles_y; ++y)
      {
        for(std::size_t x = 0; x < num_tiles_x; ++x)
          std::cout << num_particles_in_tile[static_cast<std::size_t>(y * num_tiles_x + x)] << " ";
        std::cout << std::endl;
      }
*/

      // Sort into tiles - start by storing the number of particles per tile
      for(std::size_t i = 0; i < num_tiles_x*num_tiles_y; ++i)
      {
        // Set tile at x,y,0 to the header information, i.e. number of particles currently in the tile
        cl_tile_buffer[i].s[0] = 0; // Number of particles in tile
        cl_tile_buffer[i].s[1] = 0.0; // maximum smoothing length of particles in tile
        cl_tile_buffer[i].s[2] = 0; // Data offset where particles of this tile are stored
        if(i > 0)
          // Calculate data offset
          cl_tile_buffer[i].s[2] = cl_tile_buffer[i-1].s[2] + num_particles_in_tile[i-1];
      }
      // Now store the particles themselves
      for(std::size_t i = 0; i < particles.size(); ++i)
      {
        result_vector4 particle = particles[i];
        // Find tile
        auto tile = tiles_grid_translator({{particle.s[0], particle.s[1]}});
        tiles_grid_translator.clamp_grid_index_to_edge(tile);

        // Make sure the headers are continous in memory
        std::size_t header_position = tile[1]*num_tiles_x + tile[0];
        result_vector4 header = cl_tile_buffer[header_position];

        std::size_t num_particles_already_in_tile =
            static_cast<std::size_t>(header.s[0]);

        // Store maximum smoothing length of tile
        result_scalar smoothing_length = particle.s[2];
        if(smoothing_length > header.s[1])
          cl_tile_buffer[header_position].s[1] = smoothing_length;

        std::size_t offset = static_cast<std::size_t>(header.s[2]);
        std::size_t insertion_position = offset + num_particles_already_in_tile;

        cl_particles[insertion_position] = particle;

        // increase number of particles in tile
        cl_tile_buffer[header_position].s[0] += 1.0f;
      }


      // Copy tiles and particles
      cl::Event tiles_copied_event;
      _ctx->memcpy_h2d_async<result_vector4>(tiles_buffer,
                                             cl_tile_buffer.data(),
                                             num_tiles_x*num_tiles_y,
                                             &tiles_copied_event,
                                             nullptr,1);
      cl::Event particles_copied_event;
      _ctx->memcpy_h2d_async<result_vector4>(particles_buffer,
                                             cl_particles.data(),
                                             cl_particles.size(),
                                             &particles_copied_event,
                                             nullptr,1);


      // Copy input buffers to device
      for(std::size_t k = 0; k < reconstruction_quantities.size(); ++k)
        _ctx->memcpy_h2d(quantities_buffer[k], cl_quantities[k].data(),
                         cl_quantities[k].size());

      // Apply the quantity transformation
      qcl::kernel_argument_list transformation_arguments{quantity_transformation_kernel};
      transformation_arguments.push(transformed_quantity_buffer);
      transformation_arguments.push(static_cast<cl_uint>(
                                      current_block.get_num_available_rows()));
      for(std::size_t k = 0; k < reconstruction_quantities.size(); ++k)
        transformation_arguments.push(quantities_buffer[k]);

      cl::Event evt;
      cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(
                       *quantity_transformation_kernel,
                       cl::NullRange,
                       cl::NDRange{math::make_multiple_of(_local_group_size,
                                   current_block.get_num_available_rows())},
                       cl::NDRange{_local_group_size},
                       nullptr,
                       &evt);

      qcl::check_cl_error(err, "Could not enqueue transformation kernel!");

      // Send the cavalry, load the big guns! Prepare and launch OpenCL kernel!
      qcl::kernel_argument_list arguments{_reconstruction_kernel};

      arguments.push(tiles_buffer);
      arguments.push(particles_buffer);
      arguments.push(static_cast<cl_float>(max_smoothing_length));
      arguments.push(output_buffer);
      arguments.push(cl_grid_min_corner);
      arguments.push(cl_grid_max_corner);
      arguments.push(static_cast<cl_uint>(out.get_extent_of_dimension(0)));
      arguments.push(static_cast<cl_uint>(out.get_extent_of_dimension(1)));
      arguments.push(static_cast<cl_uint>(num_tiles_x));
      arguments.push(static_cast<cl_uint>(num_tiles_y));
      //arguments.push(nullptr, sizeof(result_vector4) * _local_group_size * _local_group_size);
      //arguments.push(nullptr, sizeof(result_scalar) * _local_group_size * _local_group_size);
      arguments.push(transformed_quantity_buffer);


      evt.wait();
      std::vector<cl::Event> events_to_wait_for{
          {tiles_copied_event, particles_copied_event}};

      err = _ctx->get_command_queue().enqueueNDRangeKernel(
                                                     *_reconstruction_kernel,
                                                     cl::NullRange,
                                                     cl::NDRange(num_tiles_x * _local_group_size,
                                                                 num_tiles_y * _local_group_size),
                                                     cl::NDRange(_local_group_size, _local_group_size),
                                                     &events_to_wait_for,
                                                     &evt);
      qcl::check_cl_error(err, "Could not enqueue kernel!");
      err = evt.wait();
      qcl::check_cl_error(err, "Error while waiting for the kernel to complete.");
    };



    // Asynchronously load and process data
    io::async_for_each_block(streamer.begin_row_blocks(blocksize),
                             streamer.end_row_blocks(),
                             block_processor);

    // Retrieve result
    _ctx->memcpy_d2h(out.data(),
                     output_buffer,
                     out.get_extent_of_dimension(0) * out.get_extent_of_dimension(1));
  }



private:




  cl::NDRange get_num_work_items(std::size_t num_pix_x, std::size_t num_pix_y) const
  {
    return cl::NDRange{math::make_multiple_of(_local_group_size, num_pix_x),
                       math::make_multiple_of(_local_group_size, num_pix_y)};
  }

  const std::size_t _local_group_size = 16;
  const math::scalar _smoothing_length_scale_factor = 3.5;
  const math::scalar _additional_border_region_size = 100.0;

  qcl::device_context_ptr _ctx;
  qcl::kernel_ptr _reconstruction_kernel;

  coordinate_transformator _coordinate_transformation;
};

}

#endif
