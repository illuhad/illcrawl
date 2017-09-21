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

#include "projective_smoothing_reconstruction.hpp"

namespace illcrawl {
namespace reconstruction_backends {
namespace dm {

projective_smoothing::projective_smoothing(
                     const qcl::device_context_ptr& ctx,
                     const camera& cam,
                     math::scalar max_integration_depth,
                     const reconstruction_quantity::quantity* q,
                     std::unique_ptr<projective_smoothing_backend>
                                smoothing_backend)
  : _ctx{ctx},
    _cam{cam},
    _backend{std::move(smoothing_backend)},
    _max_integration_depth{max_integration_depth},
    _quantity{q}
{
}


std::vector<H5::DataSet>
projective_smoothing::get_required_additional_datasets() const
{
  return _backend->get_required_additional_datasets();
}

const cl::Buffer&
projective_smoothing::retrieve_results()
{
  device_scalar pixel_area =
      static_cast<device_scalar>(_cam.get_pixel_area());

  device_scalar length_conversion =
      static_cast<device_scalar>(
        _quantity->get_unit_converter().length_conversion_factor());

  device_scalar effective_dA = static_cast<device_scalar>(
        _quantity->effective_line_of_sight_integration_dA(
          pixel_area * _quantity->get_unit_converter().area_conversion_factor(),
          length_conversion * _max_integration_depth));

  // Multiply results by dA and integration length correction

  const cl::Buffer& result = _backend->retrieve_results();

  qcl::kernel_ptr scaling_kernel = _ctx->get_kernel("vector_scale");
  qcl::kernel_argument_list args{scaling_kernel};
  args.push(result);
  args.push(effective_dA);
  args.push(static_cast<cl_ulong>(this->_num_evaluation_points));

  cl::Event scaling_finished;
  cl_int err = _ctx->enqueue_ndrange_kernel(scaling_kernel,
                                            cl::NDRange{_num_evaluation_points},
                                            cl::NDRange{local_size},
                                            &scaling_finished);
  qcl::check_cl_error(err, "Could not enqueue vector_scale kernel");
  err = scaling_finished.wait();
  qcl::check_cl_error(err, "Error while waiting for the vector_scale kernel"
                           " to complete.");


  return _backend->retrieve_results();
}

std::string
projective_smoothing::get_backend_name() const
{
  return "projective_smoothing via "+_backend->get_backend_name();
}

void
projective_smoothing::init_backend(std::size_t blocksize)
{
  _backend->init_backend(blocksize);
}

void
projective_smoothing::setup_particles(const std::vector<particle>& particles,
                                      const std::vector<cl::Buffer>& additional_dataset)
{
  cl::Buffer projected_particles;
  _ctx->create_buffer<particle>(projected_particles, particles.size());
  _ctx->memcpy_h2d(projected_particles, particles.data(), particles.size());

  this->project_particles(_cam,
                          projected_particles,
                          particles.size(),
                          _max_integration_depth);

  _backend->setup_projected_particles(projected_particles,
                                      particles.size(),
                                      additional_dataset);
}

void
projective_smoothing::setup_evaluation_points(const cl::Buffer& evaluation_points,
                                              std::size_t num_points)
{
  this->project_evaluation_points(_cam, evaluation_points, num_points);
  _backend->setup_evaluation_points(evaluation_points, num_points);

  this->_num_evaluation_points = num_points;
}

void
projective_smoothing::run()
{
  this->_backend->run();
}

void
projective_smoothing::set_camera(const camera& cam)
{
  this->_cam = cam;
}

void
projective_smoothing::set_integration_depth(math::scalar depth)
{
  this->_max_integration_depth = depth;
}

void
projective_smoothing::project_evaluation_points(
                               const camera& cam,
                               const cl::Buffer& evaluation_points,
                               std::size_t num_evaluation_points) const
{
  qcl::kernel_ptr projection_kernel =
      _ctx->get_kernel("project_evaluation_points");

  qcl::kernel_argument_list args{projection_kernel};
  args.push(evaluation_points);
  args.push(static_cast<cl_ulong>(num_evaluation_points));
  args.push(math::to_device_vector4(cam.get_position()));
  args.push(math::to_device_vector4(cam.get_screen_basis_vector0()));
  args.push(math::to_device_vector4(cam.get_screen_basis_vector1()));
  args.push(math::to_device_vector4(cam.get_look_at()));

  cl::Event projection_finished;
  cl_int err = _ctx->enqueue_ndrange_kernel(projection_kernel,
                                            cl::NDRange{num_evaluation_points},
                                            cl::NDRange{local_size},
                                            &projection_finished);
  qcl::check_cl_error(err, "Could not enqueue evaluation point projection "
                           "kernel.");

  err = projection_finished.wait();
  qcl::check_cl_error(err, "Error while waiting for the evaluation point "
                           "projection kernel to finish");

}

void
projective_smoothing::project_particles(
                       const camera& cam,
                       const cl::Buffer& particles,
                       std::size_t num_particles,
                       math::scalar max_projection_distance) const
{
  qcl::kernel_ptr projection_kernel =
      _ctx->get_kernel("project_particles");

  qcl::kernel_argument_list args{projection_kernel};
  args.push(particles);
  args.push(static_cast<cl_ulong>(num_particles));
  args.push(static_cast<device_scalar>(max_projection_distance));
  args.push(math::to_device_vector4(cam.get_position()));
  args.push(math::to_device_vector4(cam.get_screen_basis_vector0()));
  args.push(math::to_device_vector4(cam.get_screen_basis_vector1()));
  args.push(math::to_device_vector4(cam.get_look_at()));

  cl::Event projection_finished;
  cl_int err = _ctx->enqueue_ndrange_kernel(projection_kernel,
                                            cl::NDRange{num_particles},
                                            cl::NDRange{local_size},
                                            &projection_finished);
  qcl::check_cl_error(err, "Could not enqueue particle projection "
                           "kernel.");

  err = projection_finished.wait();
  qcl::check_cl_error(err, "Error while waiting for the particle "
                           "projection kernel to finish");
}


}
}
}
