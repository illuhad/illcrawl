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


#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <array>
#include <cmath>

#include "math.hpp"
#include "qcl.hpp"
#include "camera.hpp"
#include "quantity.hpp"

namespace illcrawl {
namespace integration {

template<class T>
class absolute_tolerance
{
public:
  absolute_tolerance(const T& tol)
    : _tol{tol}
  {}

  inline
  T get_absolute_tolerance(const T& integration_state) const
  {
    return _tol;
  }

  inline
  T get() const
  {
    return _tol;
  }
private:
  T _tol;
};

template<class T>
class relative_tolerance
{
public:
  relative_tolerance(const T& tol)
    : _tol{tol}
  {}

  inline
  T get_absolute_tolerance(const T& integration_state) const
  {
    return _tol * integration_state;
  }

  inline
  T get() const
  {
    return _tol;
  }

private:
  T _tol;
};

template<class Tolerance_type>
struct is_relative_tolerance
{
  static constexpr bool value = false;
};

template<class T>
struct is_relative_tolerance<relative_tolerance<T>>
{
  static constexpr bool value = true;
};

/// Solves dy/dz = f(z)
template<class T,
         class Coordinate_type>
class runge_kutta_fehlberg
{
public:
  runge_kutta_fehlberg(Coordinate_type initial_position = Coordinate_type{},
                       T initial_state = T{},
                       T first_evaluation = T{},
                       Coordinate_type initial_step_size = 1.0)
    : _state{initial_state},
      _interval_start_evaluation{first_evaluation},
      _step_size{initial_step_size},
      _current_position{initial_position}
  {}

  using evaluation_coordinates = std::array<Coordinate_type, 4>;
  using integrand_values = std::array<T, 4>;
  static constexpr std::size_t required_num_evaluations = 4;

  Coordinate_type get_position() const
  {
    return _current_position;
  }

  T get_state() const
  {
    return _state;
  }

  Coordinate_type get_step_size() const
  {
    return _step_size;
  }

  void obtain_next_step_coordinates(evaluation_coordinates& next_step_coordinates) const
  {
    // k1 will be provided by the last evaluation of the previous interval
    // next_step_coordinates[0] = _current_position;

    // We do not need k2 since we are only interested in the equation
    // dy/dz =f(z) and not dy/dz=f(y,z)
    //next_step_coordinates[1] = _current_position + 1./4. * _step_size;
    next_step_coordinates[0] = _current_position + 3./8. * _step_size;
    next_step_coordinates[1] = _current_position + 12./13. * _step_size;
    next_step_coordinates[2] = _current_position + _step_size;
    next_step_coordinates[3] = _current_position + 1./2. * _step_size;
  }

  template<class Tolerance_type>
  void advance(const integrand_values& values,
               const Tolerance_type& tolerance,
               Coordinate_type integration_end)
  {
    T delta4 =
           + 25./216.    * _interval_start_evaluation
           + 1408./2565. * values[0]
           + 2197./4101. * values[1]
           - 1./5.       * values[2];

    T delta5 =
           + 16./135.      * _interval_start_evaluation
           + 6656./12825.  * values[0]
           + 28561./56430. * values[1]
           - 9./50.        * values[2]
           + 2./55.        * values[3];

    delta4 *= _step_size;
    delta5 *= _step_size;

    T estimate4 = _state + delta4;
    T estimate5 = _state + delta5;

    _current_position += _step_size;


    T s = 2.0;
    if(estimate4 != estimate5)
    {
      T error = std::abs(estimate5 - estimate4);

      T absolute_tolerance = tolerance.get_absolute_tolerance(_state / _current_position);
      s = std::pow(absolute_tolerance * _step_size / (2 * error), 1./4.);
    }

    Coordinate_type new_step_size = s * _step_size;

    if(new_step_size < minimum_stepsize)
    {
      new_step_size = minimum_stepsize;
      s = new_step_size / _step_size;
    }

    if(s < 0.95)
    {
      // Reject approximation, go back to old position
      _current_position -= _step_size;
    }
    else
    {
      // Accept approximation
      _state = estimate4;
      _interval_start_evaluation = values[2];
    }

    _step_size = new_step_size;

    if(_current_position + _step_size > integration_end)
      // The epsilon's job is to make sure that the condition
      // get_position() < integration range turns false and a integration loop
      // does not turn into an infinite loop.
      _step_size = integration_end - _current_position + epsilon;
  }

private:
  static constexpr Coordinate_type minimum_stepsize = 0.2;
  static constexpr Coordinate_type epsilon = 0.01;

  T _state;
  T _interval_start_evaluation;

  Coordinate_type _step_size;
  Coordinate_type _current_position;
};

/// Maps the scalar positions where \c parallel_runge_kutta_fehlberg
/// wants to obtain evaluations of the integrand to 3d coordinates
/// by interpreting them as z coordinates over a camera plane
/// within the x-y plane. Note that the x,y,z basis vectors get
/// supplied by the \c camera object, hence we can also use e.g.
/// rotated coordinated systems.
template<class Volumetric_reconstructor>
class parallel_pixel_integrand
{
public:

  parallel_pixel_integrand(const qcl::device_context_ptr& ctx,
                           const camera& cam,
                           Volumetric_reconstructor* reconstructor,
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

  void operator()(const cl::Buffer& required_evaluation_points_buffer,
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

private:
  static constexpr std::size_t local_size2D = 16;
  static constexpr std::size_t local_size = 256;

  qcl::device_context_ptr _ctx;
  camera _cam;

  qcl::kernel_ptr _evaluation_point_constructor;
  qcl::kernel_ptr _gather_integrand_evaluations;

  cl::Buffer _constructed_evaluation_points;

  Volumetric_reconstructor* _reconstructor;
  const reconstruction_quantity::quantity* _quantity;

};


class parallel_runge_kutta_fehlberg
{
public:

  parallel_runge_kutta_fehlberg(const qcl::device_context_ptr& ctx,
                            std::size_t num_integrators,
                            math::scalar initial_position = 0.0,
                            math::scalar initial_stepsize = 1.0,
                            math::scalar initial_integrand_evaluation = 0.0)
    : _ctx{ctx},
      _total_num_integrators{num_integrators},
      _integration_kernel{ctx->get_kernel("runge_kutta_fehlberg")},
      _num_running_integrators{0},
      _queue{ctx->get_command_queue().get()}
  {
    assert(ctx != nullptr);
    assert(num_integrators > 0);

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

  inline std::size_t get_num_running_integrators() const
  {
    return _num_running_integrators;
  }

  template<class Tolerance_type,
           class Integrand_evaluator>
  inline void advance(const Tolerance_type& tol,
                      math::scalar integration_end,
                      Integrand_evaluator f)
  {
    if(_num_running_integrators == 0)
      return;

    // First, evaluate the integrand at the required positions.
    // This is somewhat complicated, because some integrators
    // may already have finished and reached the end of the
    // integration domain. For performance reasons, we do not
    // want to evaluate these positions again - the integrand
    // evaluation is expensive and should therefore only take
    // place for the integrators which actually are still running.
    // This means that we must "compress" the array of evaluation positions
    // by removing all entries which are unneeded. We accomplish
    // this efficiently by calculating the cumulative sum of the
    // integration state (1 if integrator is still running, otherwise 0)
    // in _cumulative_num_running_integrators, such that
    // _cumulative_num_running_integrators[i] contains the index
    // in the compressed array where this integrator needs to be.
    f(_required_evaluation_points_buffer,
      _is_integrator_still_running_buffer,
      _cumulative_num_running_integrators,
      _total_num_integrators,
      _num_running_integrators,
      _evaluations_buffer);


    // Now advance the integration state
    integration_step(tol, integration_end);

    // Update the number of running integrators
    update_num_running_integrators();

  }

  const cl::Buffer& get_integration_state() const
  {
    return _integration_state_buffer;
  }

private:

  static constexpr std::size_t local_size = 256;

  template<class Tolerance_type>
  void integration_step(const Tolerance_type& tol,
                       math::scalar integration_end)
  {
    qcl::kernel_argument_list args{_integration_kernel};
    args.push(_integration_state_buffer);
    args.push(_current_position_buffer);
    args.push(_current_step_size_buffer);
    args.push(_evaluations_buffer);
    args.push(_range_begin_evaluation_buffer);
    args.push(static_cast<cl_int>(_total_num_integrators));
    args.push(static_cast<device_scalar>(tol.get()));

    bool is_rel_tol = is_relative_tolerance<Tolerance_type>::value;
    args.push(static_cast<cl_int>(is_rel_tol));

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

  void update_num_running_integrators()
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

  qcl::device_context_ptr _ctx;
  std::size_t _total_num_integrators;

  qcl::kernel_ptr _integration_kernel;
  std::size_t _num_running_integrators;

  cl::Buffer _integration_state_buffer;
  cl::Buffer _current_position_buffer;
  cl::Buffer _current_step_size_buffer;
  cl::Buffer _evaluations_buffer;
  cl::Buffer _range_begin_evaluation_buffer;

  cl::Buffer _required_evaluation_points_buffer; // of type evaluation_points = float4
  cl::Buffer _is_integrator_still_running_buffer; // of type int

  cl::Buffer _cumulative_num_running_integrators;

  boost::compute::command_queue _queue;


};

}
}

#endif
