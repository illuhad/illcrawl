
#include <iostream>

#include "async_io.hpp"
#include "fits.hpp"
#include "hdf5_io.hpp"
#include "particle_distribution.hpp"
#include "qcl.hpp"
#include "reconstruction.hpp"
#include "volumetric_reconstruction.hpp"

void usage()
{
  std::cout << "Usage: illcrawl <Path to HDF5 file>" << std::endl;
}

using result_scalar = illcrawl::smoothed_quantity_reconstruction2D::result_scalar;
using render_result = illcrawl::util::multi_array<result_scalar>;

void render_view3d(const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction)
{

  std::size_t resolution = 2048;
  std::size_t num_frames = 400;
  illcrawl::util::multi_array<result_scalar> result_data_cube{resolution, resolution, num_frames};
  illcrawl::util::multi_array<result_scalar> result;

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
  illcrawl::util::fits<result_scalar> result_file{"illcrawl_3d_render.fits"};
  result_file.save(result_data_cube);
}

void render_quantity(const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction,
                   const illcrawl::reconstruction_quantity::quantity& rendered_quantity,
                   illcrawl::util::multi_array<result_scalar>& result)
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
  illcrawl::util::multi_array<result_scalar> result;
  render_quantity(center, loader, reconstruction, rendered_quantity, result);

  illcrawl::util::fits<result_scalar> result_file{filename};
  result_file.save(result);
}

void render_luminosity_weighted_temperature(
                   const illcrawl::math::vector3& center,
                   const illcrawl::io::illustris_gas_data_loader& loader,
                   illcrawl::smoothed_quantity_reconstruction2D& reconstruction,
                   const illcrawl::util::multi_array<result_scalar>& xray_emission,
                   const std::string& filename = "illcrawl_render.fits")
{
  auto luminosity_weighted_temperature = std::make_shared<
      illcrawl::reconstruction_quantity::luminosity_weighted_temperature>(&loader);


  illcrawl::util::multi_array<result_scalar> result;
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

  illcrawl::util::fits<result_scalar> result_file{filename};
  result_file.save(result);

}

int main(int argc, char** argv)
{

  if (argc != 2)
  {
    usage();

    return -1;
  }
  std::string data_file = argv[1];

  qcl::environment env;

  const cl::Platform& plat =
      env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});

  qcl::global_context_ptr global_ctx =
      env.create_global_context(plat, CL_DEVICE_TYPE_GPU);

  if (global_ctx->get_num_devices() == 0)
  {
    std::cout << "No OpenCL GPU devices found!" << std::endl;
    return -1;
  }
  else
  {
    for (std::size_t i = 0; i < global_ctx->get_num_devices(); ++i)
    {
      std::cout << "Device " << i << ":" << std::endl;
      std::cout << "   Name: " << global_ctx->device(i)->get_device_name()
                << std::endl;

      std::string extensions;
      global_ctx->device(i)->get_supported_extensions(extensions);
      std::cout << "   Capabilities: " << extensions << std::endl;
    }
  }

  global_ctx->global_register_source_file("reconstruction.cl",
                                          {"image_tile_based_reconstruction2D",
                                           "volumetric_reconstruction",
                                           "finalize_volumetric_reconstruction"});
  global_ctx->global_register_source_file("interpolation_tree.cl",
                                          {"tree_interpolation"});

  global_ctx->global_register_source_file("quantities.cl",
                                          // Kernels inside quantities.cl
                                          {
                                            "luminosity_weighted_temperature",
                                            "xray_emission",
                                            "identity",
                                            "mean_temperature"
                                          });

  qcl::device_context_ptr ctx = global_ctx->device();

  illcrawl::io::illustris_gas_data_loader loader{data_file};

  illcrawl::particle_distribution distribution{loader.get_coordinates()};

  illcrawl::math::vector3 distribution_center =
      distribution.get_distribution_center();
  illcrawl::math::vector3 distribution_size =
      distribution.get_distribution_size();


  std::cout << "Distribution center: " << distribution_center[0] << ", "
            << distribution_center[1] << ", "
            << distribution_center[2]
            << std::endl;
  std::cout << "Distribution size: " << distribution_size[0] << "x"
            << distribution_size[1] << "x" << distribution_size[2] << std::endl;


  auto luminosity_weighted_temperature = std::make_shared<
      illcrawl::reconstruction_quantity::luminosity_weighted_temperature>(&loader);

  auto xray_emission = std::make_shared<
      illcrawl::reconstruction_quantity::xray_emission>(&loader);

  auto weights =
      std::make_shared<illcrawl::reconstruction_quantity::interpolation_weight>(&loader);

  auto mean_temperature =
      std::make_shared<illcrawl::reconstruction_quantity::mean_temperature>(&loader);

  illcrawl::math::vector3 center = {{0.0, distribution_center[1], distribution_center[2]}};

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

  /*
  illcrawl::volumetric_tomography tomography{cam};
  integrator.create_tomographic_cube(reconstruction, *xray_emission, 10.0, result);
  */

  illcrawl::math::vector3 periodic_wraparound {{75000.0, 75000.0, 75000.0}};
  illcrawl::math::vector3 volume_size = {{2000.0, 2000.0, 2000.0}};
  illcrawl::math::vector3 camera_look_at = {{0., 0., 1.}};
  illcrawl::volume_cutout total_render_volume{center, volume_size, periodic_wraparound};

  illcrawl::math::vector3 camera_pos = {{0.0, distribution_center[1], distribution_center[2]-100.0}};
  illcrawl::camera cam{camera_pos, camera_look_at, 0.0, 1500.0, 1024, 1024};

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


  illcrawl::volumetric_tree_reconstruction reconstructor{
    ctx, total_render_volume, loader.get_coordinates(), 7000000, 0.9};

  //illcrawl::volumetric_slice<illcrawl::volumetric_tree_reconstruction> slice{cam};
  //slice.create_slice(reconstructor, *xray_emission, result, 0);

  //illcrawl::volumetric_tomography<illcrawl::volumetric_tree_reconstruction> tomography{cam};
  //tomography.create_tomographic_cube(reconstructor, *xray_emission, 1000.0, result);

  illcrawl::volumetric_integration<illcrawl::volumetric_tree_reconstruction> integrator{cam};
  integrator.create_projection(reconstructor, *xray_emission, 1000.0, 1000.0, result);

  illcrawl::util::fits<result_scalar> result_file{"illcrawl_render.fits"};
  result_file.save(result);

  return 0;
}
