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


#include <iostream>

#include "async_io.hpp"
#include "fits.hpp"
#include "hdf5_io.hpp"
#include "particle_distribution.hpp"
#include "qcl.hpp"
#include "reconstruction.hpp"
#include "volumetric_reconstruction.hpp"
#include "environment.hpp"
#include "master_ostream.hpp"
#include "tree_ostream.hpp"
#include "partitioner.hpp"
#include "animation.hpp"

void usage(std::ostream& ostr)
{
  ostr << "Usage: illcrawl <Path to HDF5 file>" << std::endl;
}

using illcrawl::device_scalar;
using render_result = illcrawl::util::multi_array<device_scalar>;

void render_view3d(const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction)
{

  std::size_t resolution = 2048;
  std::size_t num_frames = 400;
  illcrawl::util::multi_array<device_scalar> result_data_cube{resolution, resolution, num_frames};
  illcrawl::util::multi_array<device_scalar> result;

  illcrawl::math::vector3 rotation_axis = {{0, 1, 0}};

  for(std::size_t i = 0; i < num_frames; ++i)
  {
    illcrawl::math::scalar angle = 2.*M_PI * static_cast<illcrawl::math::scalar>(i) / num_frames;

    std::cout << "Angle = " << angle << std::endl;
    illcrawl::math::matrix3x3 rotation_matrix;
    illcrawl::math::matrix_create_rotation_matrix(&rotation_matrix,
                                                  rotation_axis,
                                                  angle);
    auto coordinate_transformation =
        [&](const illcrawl::math::vector3& v) -> illcrawl::math::vector3
    {
      using namespace illcrawl;
      illcrawl::math::vector3 v_prime = v - center;
      v_prime = illcrawl::math::matrix_vector_mult(rotation_matrix, v_prime);
      return v_prime + center;
    };

    auto xray_emission = std::make_shared<
        illcrawl::reconstruction_quantity::xray_emission>(&loader);

    reconstruction.set_coordinate_transformation(coordinate_transformation);
    reconstruction.run(
        *xray_emission,
        loader.get_coordinates(), loader.get_smoothing_length(), loader.get_volume(),
        center, 2000, 2000, resolution,
        {{75000.0, 75000.0, 75000.0}}, result);

    for(std::size_t x = 0; x < result.get_extent_of_dimension(0); ++x)
      for(std::size_t y = 0; y < result.get_extent_of_dimension(1); ++y)
      {
        std::size_t idx2 [] = {x,y};
        std::size_t idx3 [] = {x,y,i};
        result_data_cube[idx3] = result[idx2];
      }
  }
  illcrawl::util::fits<device_scalar> result_file{"illcrawl_3d_render.fits"};
  result_file.save(result_data_cube);
}

void render_quantity(const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction,
                   const illcrawl::reconstruction_quantity::quantity& rendered_quantity,
                   illcrawl::util::multi_array<device_scalar>& result)
{

  reconstruction.run(
      rendered_quantity,
      loader.get_coordinates(), loader.get_smoothing_length(), loader.get_volume(),
      center, 2000, 2000, 2048,
      {{75000.0, 75000.0, 75000.0}}, result);
}

void render_quantity(const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction,
                   const illcrawl::reconstruction_quantity::quantity& rendered_quantity,
                   const std::string& filename = "illcrawl_render.fits")
{
  illcrawl::util::multi_array<device_scalar> result;
  render_quantity(center, loader, reconstruction, rendered_quantity, result);

  illcrawl::util::fits<device_scalar> result_file{filename};
  result_file.save(result);
}

void render_luminosity_weighted_temperature(
                   const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction,
                   const illcrawl::util::multi_array<device_scalar>& xray_emission,
                   const std::string& filename = "illcrawl_render.fits")
{
  auto luminosity_weighted_temperature = std::make_shared<
      illcrawl::reconstruction_quantity::luminosity_weighted_temperature>(&loader);


  illcrawl::util::multi_array<device_scalar> result;
  render_quantity(center,
                  loader,
                  reconstruction,
                  *luminosity_weighted_temperature,
                  result);

  assert(xray_emission.get_dimension() == result.get_dimension());
  assert(xray_emission.get_extent_of_dimension(0) ==
         result.get_extent_of_dimension(0));
  assert(xray_emission.get_extent_of_dimension(1) ==
         result.get_extent_of_dimension(1));

  for(std::size_t i = 0; i < result.get_extent_of_dimension(0); ++i)
    for(std::size_t j = 0; j < result.get_extent_of_dimension(1); ++j)
    {
      std::size_t idx [] = {i,j};
      result[idx] /= xray_emission[idx];
    }

  illcrawl::util::fits<device_scalar> result_file{filename};
  result_file.save(result);

}

int main(int argc, char** argv)
{
  illcrawl::environment env{argc, argv};
  illcrawl::util::master_ostream master_cout{
    std::cout,
    env.get_communicator(),
    env.get_master_rank()
  };

  if (argc != 2)
  {
    usage(master_cout);

    return -1;
  }
  std::string data_file = argv[1];

  master_cout << "Detected system configuration:" << std::endl;
  illcrawl::util::tree_formatter tree_output{"Root"};
  for(const auto& node : env.get_node_rank_map())
  {
    std::string node_entry_name = node.first;
    illcrawl::util::tree_formatter::node node_entry;
    for(int rank = 0; rank < static_cast<int>(node.second.size()); ++rank)
    {
      int global_rank = node.second[rank];
      std::string process_entry_name = "Local rank "
          +std::to_string(rank)
          +" (Global rank "+std::to_string(global_rank)
          +")";

      illcrawl::util::tree_formatter::node process_entry;
      process_entry.append_content("Using device: " + env.get_device_names()[global_rank]+"\n");
      process_entry.append_content("Device capabilities: " + env.get_device_extensions()[global_rank]+"\n");

      node_entry.add_node(process_entry_name, process_entry);
    }
    tree_output.get_root().add_node(node_entry_name, node_entry);
  }
  master_cout << tree_output << std::endl;


  qcl::device_context_ptr ctx = env.get_compute_device();

  illcrawl::io::illustris_gas_data_loader loader{data_file};

  illcrawl::math::vector3 periodic_wraparound {{75000.0, 75000.0, 75000.0}};
  illcrawl::particle_distribution distribution{
    loader.get_coordinates(),
    periodic_wraparound
  };

  illcrawl::math::vector3 distribution_center =
      distribution.get_extent_center();
  illcrawl::math::vector3 distribution_size =
      distribution.get_distribution_size();


  master_cout << "Distribution center: " << distribution_center[0] << ", "
              << distribution_center[1] << ", "
              << distribution_center[2]
              << std::endl;
  master_cout << "Distribution size: " << distribution_size[0] << "x"
              << distribution_size[1] << "x" << distribution_size[2] << std::endl;


  auto luminosity_weighted_temperature = std::make_shared<
      illcrawl::reconstruction_quantity::luminosity_weighted_temperature>(&loader);

  auto xray_emission = std::make_shared<
      illcrawl::reconstruction_quantity::xray_emission>(&loader);

  auto weights =
      std::make_shared<illcrawl::reconstruction_quantity::interpolation_weight>(&loader);

  auto mean_temperature =
      std::make_shared<illcrawl::reconstruction_quantity::mean_temperature>(&loader);

  auto chandra_xray_emission=
      std::make_shared<illcrawl::reconstruction_quantity::chandra_xray_emission>(&loader, ctx);

  illcrawl::math::vector3 center = distribution_center;

  //illcrawl::smoothed_quantity_reconstruction2D reconstruction{ctx};

  //render_view3d(center,
  //              loader, reconstruction);

  /*
  render_result xray_emission_result;
  render_quantity(center,
                  loader,
                  reconstruction,
                  *xray_emission,
                  xray_emission_result);


  render_luminosity_weighted_temperature(center,
                                         loader,
                                         reconstruction,
                                         xray_emission_result);


  render_quantity(center,
                  loader,
                  reconstruction,
                  *xray_emission,
                  "illcrawl_render_emission.fits");
  */


  illcrawl::math::vector3 volume_size = distribution_size;
  illcrawl::math::vector3 camera_look_at = {{0., 0., 1.}};
  illcrawl::volume_cutout total_render_volume{center, volume_size, periodic_wraparound};

  illcrawl::math::vector3 camera_pos = center;
  illcrawl::math::scalar camera_distance = 1000.0;
  camera_pos[2] -= camera_distance;
  illcrawl::camera cam{camera_pos, camera_look_at, 0.0, distribution_size[0], 1024, 1024};

  render_result result;
  /*
  illcrawl::volumetric_nn8_reconstruction reconstruction{
        ctx,
        total_render_volume,
        loader.get_coordinates(),
        loader.get_smoothing_length()
  };

  illcrawl::volumetric_integration<illcrawl::volumetric_nn8_reconstruction> integrator{cam};
  integrator.create_projection(reconstruction, *xray_emission, 10.0, 1000.0, result);

  illcrawl::util::fits<result_scalar> result_file{"illcrawl_render.fits"};
  result_file.save(result); */


  //illcrawl::volumetric_tree_reconstruction reconstructor{
  //  ctx, total_render_volume, loader.get_coordinates(), 7000000, 0.9};


  illcrawl::volumetric_nn8_reconstruction reconstructor{
    ctx, total_render_volume, loader.get_coordinates(), loader.get_smoothing_length(), 7000000};

  //illcrawl::volumetric_slice<illcrawl::volumetric_nn8_reconstruction> slice{cam};
  //slice.create_slice(reconstructor, *xray_emission, result, 0);

  //illcrawl::distributed_volumetric_tomography<illcrawl::volumetric_nn8_reconstruction,
  //                                            illcrawl::uniform_work_partitioner>
  //    tomography{illcrawl::uniform_work_partitioner{env.get_communicator()}, cam};
  // Create tomography on the master rank
  //tomography.create_tomographic_cube(reconstructor, *chandra_xray_emission, 1000.0, result);

  illcrawl::integration::absolute_tolerance<illcrawl::math::scalar> tol{10.0};


  illcrawl::camera_movement::dual_axis_rotation_around_point
  camera_mover{
    distribution_center,
    illcrawl::math::vector3{{0,0,1}}, // initial phi axis
    illcrawl::math::vector3{{1,0,0}}, // theta axis
    cam,
    10.0 * 360.0, // phi range
    360.0  // theta range
  };

  illcrawl::animation_frame::integrated_projection
  <
    illcrawl::volumetric_nn8_reconstruction,
    illcrawl::integration::absolute_tolerance<illcrawl::math::scalar>
  > frame_renderer {
    env.get_compute_device(),
    *chandra_xray_emission, // reconstructed quantity
    2.0 * camera_distance, // integration depth
    tol, // integration tolerance
    reconstructor // reconstruction engine
  };

  illcrawl::distributed_animation<illcrawl::uniform_work_partitioner>
  animation{
    illcrawl::uniform_work_partitioner{env.get_communicator()},
    frame_renderer,
    camera_mover,
    cam
  };

  animation(1000, result);

  //illcrawl::volumetric_integration<illcrawl::volumetric_nn8_reconstruction> integrator{ctx, cam};
  //integrator.parallel_create_projection(reconstructor, *chandra_xray_emission, 1000.0,
  //                             tol, result);
  
  /*render_result temperature_result;
  illcrawl::volumetric_nn8_reconstruction reconstructor_b{
    ctx, total_render_volume, loader.get_coordinates(), loader.get_smoothing_length(), 7000000};
  integrator.create_projection(reconstructor_b, *luminosity_weighted_temperature, 1000.0,
                               tol, temperature_result);
  
  for(std::size_t i = 0; i < temperature_result.get_num_elements(); ++i)
    result.data()[i] = temperature_result.data()[i] / result.data()[i];*/

  master_cout << "Saving result..." << std::endl;

  illcrawl::util::distributed_fits_slices<illcrawl::uniform_work_partitioner, device_scalar>
      result_file{animation.get_partitioning(), "illcrawl_render.fits"};

  result_file.save(result);

  return 0;
}
