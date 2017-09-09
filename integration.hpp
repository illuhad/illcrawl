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
#include "reconstructing_data_crawler.hpp"

namespace illcrawl {
namespace integration {

class tolerance
{
public:
  tolerance();
  tolerance(math::scalar abs_tol,
            math::scalar rel_tol);

  math::scalar get_absolute_tolerance() const;
  math::scalar get_relative_tolerance() const;
private:
  math::scalar _abs_tol;
  math::scalar _rel_tol;
};


/// Maps the scalar positions where \c parallel_runge_kutta_fehlberg
/// wants to obtain evaluations of the integrand to 3d coordinates
/// by interpreting them as z coordinates over a camera plane
/// within the x-y plane. Note that the x,y,z basis vectors get
/// supplied by the \c camera object, hence we can also use e.g.
/// rotated coordinated systems.

class parallel_pixel_integrand
{
public:

  parallel_pixel_integrand(const qcl::device_context_ptr& ctx,
                           const camera& cam,
                           reconstructing_data_crawler* reconstructor,
                           const reconstruction_quantity::quantity* q);

  void operator()(const cl::Buffer& required_evaluation_points_buffer,
                  const cl::Buffer& is_integrator_still_running_buffer,
                  const cl::Buffer& cumulative_num_running_integrators,
                  std::size_t total_num_integrators,
                  std::size_t num_running_integrators,
                  const cl::Buffer& evaluations_buffer) const;

private:
  static constexpr std::size_t local_size2D = 16;
  static constexpr std::size_t local_size = 256;

  qcl::device_context_ptr _ctx;
  camera _cam;

  qcl::kernel_ptr _evaluation_point_constructor;
  qcl::kernel_ptr _gather_integrand_evaluations;

  cl::Buffer _constructed_evaluation_points;

  reconstructing_data_crawler* _reconstructor;
  const reconstruction_quantity::quantity* _quantity;

};


class parallel_runge_kutta_fehlberg
{
public:

  parallel_runge_kutta_fehlberg(const qcl::device_context_ptr& ctx,
                            std::size_t num_integrators,
                            math::scalar initial_stepsize = 1.0,
                            math::scalar initial_integrand_evaluation = 0.0);


  template<class Integrand_evaluator>
  inline void advance(const tolerance& tol,
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

  std::size_t get_num_running_integrators() const;
  const cl::Buffer& get_integration_state() const;

private:

  static constexpr std::size_t local_size = 256;

  void integration_step(const tolerance& tol,
                       math::scalar integration_end);

  void update_num_running_integrators();

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
