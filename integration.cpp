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

#include <cassert>
#include "integration.hpp"

namespace illcrawl {
namespace integration {

/*************** Implementation of tolerance *******************/

tolerance::tolerance()
  : _abs_tol{0.0},
    _rel_tol{0.0}
{}

tolerance::tolerance(math::scalar abs_tol,
                     math::scalar rel_tol)
  : _abs_tol{abs_tol},
    _rel_tol{rel_tol}
{
  assert(_abs_tol >= 0.0);
  assert(_rel_tol >= 0.0);
}

math::scalar
tolerance::get_absolute_tolerance() const
{
  return _abs_tol;
}

math::scalar
tolerance::get_relative_tolerance() const
{
  return _rel_tol;
}

/*************** Implementation of parallel_pixel_integrand ************/


parallel_pixel_integrand::parallel_pixel_integrand(
                         const qcl::device_context_ptr& ctx,
                         const camera& cam,
                         reconstructing_data_crawler* reconstructor,
                         const reconstruction_quantity::quantity* q)
  : _ctx{ctx},
    _cam{cam},
    _evaluation_point_constructor{
      ctx->get_kernel("construct_evaluation_points_over_camera_plane")
    },
    _gather_integrand_evaluations{
      ctx->get_kernel("gather_integrand_evaluations")
     },
    _reconstructor{reconstructor},
    _quantity{q}
{
  assert(ctx != nullptr);
  assert(reconstructor != nullptr);
  assert(q != nullptr);

  _ctx->create_buffer<device_vector4>(_constructed_evaluation_points,
                                      CL_MEM_READ_WRITE,
                                      4 * cam.get_num_pixels(0) * cam.get_num_pixels(1));
}

void
parallel_pixel_integrand::operator()(const cl::Buffer& required_evaluation_points_buffer,
                                     const cl::Buffer& is_integrator_still_running_buffer,
                                     const cl::Buffer& cumulative_num_running_integrators,
                                     std::size_t total_num_integrators,
                                     std::size_t num_running_integrators,
                                     const cl::Buffer& evaluations_buffer) const
{
  assert(_cam.get_num_pixels(0) * _cam.get_num_pixels(1) == total_num_integrators);

  qcl::kernel_argument_list construction_args{_evaluation_point_constructor};
  construction_args.push(required_evaluation_points_buffer);
  construction_args.push(cumulative_num_running_integrators);
  construction_args.push(is_integrator_still_running_buffer);

  device_vector4 camera_look_at = math::to_device_vector4(_cam.get_look_at());
  device_vector4 camera_x_basis = math::to_device_vector4(_cam.get_screen_basis_vector0());
  device_vector4 camera_y_basis = math::to_device_vector4(_cam.get_screen_basis_vector1());
  device_vector4 camera_plane_min_position
      = math::to_device_vector4(_cam.get_screen_min_position());

  construction_args.push(camera_look_at);
  construction_args.push(camera_x_basis);
  construction_args.push(camera_y_basis);
  construction_args.push(camera_plane_min_position);
  construction_args.push(static_cast<cl_int>(_cam.get_num_pixels(0)));
  construction_args.push(static_cast<cl_int>(_cam.get_num_pixels(1)));
  construction_args.push(static_cast<device_scalar>(_cam.get_pixel_size()));

  construction_args.push(_constructed_evaluation_points);

  cl::Event evaluation_points_constructed;
  cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(*_evaluation_point_constructor,
                                                              cl::NullRange,
                                                              cl::NDRange{math::make_multiple_of(local_size2D, _cam.get_num_pixels(0)),
                                                                          math::make_multiple_of(local_size2D, _cam.get_num_pixels(1))},
                                                              cl::NDRange{local_size2D, local_size2D},
                                                              nullptr,
                                                              &evaluation_points_constructed);
  qcl::check_cl_error(err, "Error while constructing 3D evaluation point coordinates over the camera plane.");

  // Execute reconstructor - we need to evaluate the integrand
  // at 4 points for each integrator.

  err = evaluation_points_constructed.wait();
  qcl::check_cl_error(err, "Error while waiting for the evaluation point constructor kernel to finish.");

  _reconstructor->run(_constructed_evaluation_points,
                      4 * num_running_integrators,
                      *_quantity);

  // Map reconstruction results back into the integrand evaluation buffer

  qcl::kernel_argument_list gather_args{_gather_integrand_evaluations};
  gather_args.push(_reconstructor->get_reconstruction());
  gather_args.push(cumulative_num_running_integrators);
  gather_args.push(is_integrator_still_running_buffer);
  gather_args.push(static_cast<cl_int>(total_num_integrators));
  gather_args.push(evaluations_buffer);

  cl::Event gather_evaluations_finished;
  err = _ctx->get_command_queue().enqueueNDRangeKernel(*_gather_integrand_evaluations,
                                                       cl::NullRange,
                                                       cl::NDRange{math::make_multiple_of(local_size, total_num_integrators)},
                                                       cl::NDRange{local_size},
                                                       nullptr,
                                                       &gather_evaluations_finished);

  qcl::check_cl_error(err, "Error while gathering the integrand evaluations");
  err = gather_evaluations_finished.wait();
  qcl::check_cl_error(err, "Error while waiting for the evaluation gathering kernel to finish.");

}

/************* Implementation of parallel_runge_kutta_fehlberg ***********/



parallel_runge_kutta_fehlberg::parallel_runge_kutta_fehlberg(
                              const qcl::device_context_ptr& ctx,
                              std::size_t num_integrators,
                              math::scalar initial_stepsize,
                              math::scalar initial_integrand_evaluation)
  : _ctx{ctx},
    _total_num_integrators{num_integrators},
    _integration_kernel{ctx->get_kernel("runge_kutta_fehlberg")},
    _num_running_integrators{0},
    _queue{ctx->get_command_queue().get()}
{
  assert(ctx != nullptr);
  assert(num_integrators > 0);

  math::scalar initial_position = 0.0;

  _ctx->create_buffer<device_scalar>(_integration_state_buffer,
                                     CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<device_scalar>(_current_position_buffer,
                                     CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<device_scalar>(_current_step_size_buffer,
                                     CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<device_vector4>(_evaluations_buffer,
                                      CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<device_scalar>(_range_begin_evaluation_buffer,
                                     CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<device_vector4>(_required_evaluation_points_buffer,
                                      CL_MEM_READ_WRITE, num_integrators);
  _ctx->create_buffer<cl_int>(_is_integrator_still_running_buffer,
                              CL_MEM_READ_WRITE, num_integrators);

  _ctx->create_buffer<cl_int>(_cumulative_num_running_integrators,
                              CL_MEM_READ_WRITE, num_integrators);

  boost::compute::fill(qcl::create_buffer_iterator<device_scalar>(_integration_state_buffer, 0),
                       qcl::create_buffer_iterator<device_scalar>(_integration_state_buffer, num_integrators),
                       0.0f,
                       _queue);
  boost::compute::fill(qcl::create_buffer_iterator<device_scalar>(_current_position_buffer, 0),
                       qcl::create_buffer_iterator<device_scalar>(_current_position_buffer, num_integrators),
                       static_cast<device_scalar>(initial_position),
                       _queue);
  boost::compute::fill(qcl::create_buffer_iterator<device_scalar>(_current_step_size_buffer, 0),
                       qcl::create_buffer_iterator<device_scalar>(_current_step_size_buffer, num_integrators),
                       static_cast<device_scalar>(initial_stepsize),
                       _queue);
  boost::compute::fill(qcl::create_buffer_iterator<boost_device_vector4>(_evaluations_buffer, 0),
                       qcl::create_buffer_iterator<boost_device_vector4>(_evaluations_buffer, num_integrators),
                       boost_device_vector4{0.0f, 0.0f, 0.0f, 0.0f},
                       _queue);
  boost::compute::fill(qcl::create_buffer_iterator<device_scalar>(_range_begin_evaluation_buffer, 0),
                       qcl::create_buffer_iterator<device_scalar>(_range_begin_evaluation_buffer, num_integrators),
                       static_cast<device_scalar>(initial_integrand_evaluation),
                       _queue);
  boost::compute::fill(qcl::create_buffer_iterator<int>(_is_integrator_still_running_buffer, 0),
                       qcl::create_buffer_iterator<int>(_is_integrator_still_running_buffer, num_integrators),
                       1,
                       _queue);

  boost_device_vector4 first_required_evaluations{
    static_cast<device_scalar>(initial_position + 3./8. * initial_stepsize),
        static_cast<device_scalar>(initial_position + 12./13. * initial_stepsize),
        static_cast<device_scalar>(initial_position + initial_stepsize),
        static_cast<device_scalar>(initial_position + 0.5 * initial_stepsize)
  };

  boost::compute::fill(qcl::create_buffer_iterator<boost_device_vector4>(_required_evaluation_points_buffer, 0),
                       qcl::create_buffer_iterator<boost_device_vector4>(_required_evaluation_points_buffer, num_integrators),
                       first_required_evaluations,
                       _queue);

  update_num_running_integrators();
}

std::size_t
parallel_runge_kutta_fehlberg::get_num_running_integrators() const
{
  return _num_running_integrators;
}

const cl::Buffer&
parallel_runge_kutta_fehlberg::get_integration_state() const
{
  return _integration_state_buffer;
}


void
parallel_runge_kutta_fehlberg::integration_step(const tolerance& tol,
                                                math::scalar integration_end)
{
  qcl::kernel_argument_list args{_integration_kernel};
  args.push(_integration_state_buffer);
  args.push(_current_position_buffer);
  args.push(_current_step_size_buffer);
  args.push(_evaluations_buffer);
  args.push(_range_begin_evaluation_buffer);
  args.push(static_cast<cl_int>(_total_num_integrators));
  args.push(static_cast<device_scalar>(tol.get_absolute_tolerance()));
  args.push(static_cast<device_scalar>(tol.get_relative_tolerance()));

  args.push(static_cast<device_scalar>(integration_end));

  args.push(_required_evaluation_points_buffer);
  args.push(_is_integrator_still_running_buffer);

  cl::Event integration_finished;
  cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(*_integration_kernel,
                                                              cl::NullRange,
                                                              cl::NDRange{math::make_multiple_of(local_size, _total_num_integrators)},
                                                              cl::NDRange{local_size},
                                                              nullptr,
                                                              &integration_finished);
  qcl::check_cl_error(err, "Could not enqueue integration kernel.");
  err = integration_finished.wait();
  qcl::check_cl_error(err, "Error while waiting for the integration kernel to finish.");
}

void
parallel_runge_kutta_fehlberg::update_num_running_integrators()
{
  auto begin = qcl::create_buffer_iterator<int>(_is_integrator_still_running_buffer, 0);
  auto end = qcl::create_buffer_iterator<int>(_is_integrator_still_running_buffer, _total_num_integrators);

  auto output = qcl::create_buffer_iterator<int>(_cumulative_num_running_integrators, 0);

  boost::compute::exclusive_scan(begin, end, output, _queue);

  cl_int is_last_integrator_running = 0;
  cl_int exclusive_sum = 0;

  _ctx->memcpy_d2h<cl_int>(&exclusive_sum,
                           _cumulative_num_running_integrators,
                           _total_num_integrators - 1,
                           _total_num_integrators, 0);

  _ctx->memcpy_d2h<cl_int>(&is_last_integrator_running,
                           _is_integrator_still_running_buffer,
                           _total_num_integrators - 1,
                           _total_num_integrators, 0);

  assert(is_last_integrator_running == 0
         || is_last_integrator_running == 1);

  this->_num_running_integrators =
      static_cast<std::size_t>(exclusive_sum + is_last_integrator_running);
}



}
}
