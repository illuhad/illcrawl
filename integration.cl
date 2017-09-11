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


#ifndef INTEGRATION_CL
#define INTEGRATION_CL

#include "types.cl"
typedef vector4 rkf_integrand_values;
typedef vector4 rkf_evaluation_points;


// epsilon defines an additional margin that is added
// on the position when we have reached the integration
// end. In order to avoid the code rejecting the last
// step, it must be smaller than the minimum step size.
#define EPSILON 0.01f
#define MINIMUM_STEPSIZE 0.2f
#define MAXIMUM_STEPSIZE 100.f

rkf_evaluation_points rkf_generate_evaluation_points(scalar current_position,
                                                     scalar current_step_size)
{
  rkf_evaluation_points result;
  result.s0 = current_position + 3.f / 8.f * current_step_size;
  result.s1 = current_position + 12.f/13.f * current_step_size;
  result.s2 = current_position +             current_step_size;
  result.s3 = current_position + 0.5f      * current_step_size;
  return result;
}

void rkf_advance(scalar* integration_state,
                 scalar* current_position,
                 scalar* current_step_size,
                 scalar* range_begin_evaluation,
                 scalar absolute_tolerance,
                 scalar relative_tolerance,
                 scalar integration_end,
                 rkf_integrand_values evaluations,
                 rkf_evaluation_points* next_evaluation_points,
                 scalar epsilon,
                 scalar min_stepsize,
                 scalar max_stepsize)
{

  scalar delta4 =
       + 25.f/216.f    * (*range_begin_evaluation)
       + 1408.f/2565.f * evaluations.s0
       + 2197.f/4101.f * evaluations.s1
       - 1.f/5.f       * evaluations.s2;

  scalar delta5 =
       + 16.f/135.f      * (*range_begin_evaluation)
       + 6656.f/12825.f  * evaluations.s0
       + 28561.f/56430.f * evaluations.s1
       - 9.f/50.f        * evaluations.s2
       + 2.f/55.f        * evaluations.s3;

  delta4 *= (*current_step_size);
  delta5 *= (*current_step_size);

  scalar estimate4 = *integration_state + delta4;

  scalar old_position = *current_position;
  *current_position += *current_step_size;

  scalar s = 2.0f;
  if(delta4 != delta5)
  {
    scalar error = fabs(delta5 - delta4);

    scalar scaled_relative_tolerance = fabs(*integration_state / (*current_position) * relative_tolerance);

    scalar overall_tolerance = fmax(absolute_tolerance, scaled_relative_tolerance);

    s = pow(overall_tolerance * (*current_step_size) / (2.f * error), 0.25f);
  }

  scalar new_step_size = s * (*current_step_size);

  if(new_step_size < min_stepsize)
  {
    new_step_size = min_stepsize;
  }
  else if(new_step_size > max_stepsize)
  {
    new_step_size = max_stepsize;
  }

  // Correct s in case we have reached the min/max stepsize
  s = new_step_size / (*current_step_size);

  if(s < 0.95f)
  {
    // Reject approximation, go back to old position
    *current_position = old_position;
  }
  else
  {
    // Accept approximation
    *integration_state = estimate4;
    *range_begin_evaluation = evaluations.s2;
  }

  *current_step_size = new_step_size;

  if(*current_position + (*current_step_size) >= integration_end)
    // The epsilon's job is to make sure that the condition
    // position < integration range turns false and a integration loop
    // does not turn into an infinite loop.
    *current_step_size = integration_end - *current_position + epsilon;

  *next_evaluation_points = rkf_generate_evaluation_points(*current_position, *current_step_size);
}





__kernel void runge_kutta_fehlberg(__global scalar* integration_state,
                                   __global scalar* current_position,
                                   __global scalar* current_step_size,
                                   __global rkf_integrand_values* evaluations,
                                   __global scalar* range_begin_evaluation,
                                   int num_integrators,
                                   scalar absolute_tolerance,
                                   scalar relative_tolerance,
                                   scalar integration_end,
                                   __global rkf_evaluation_points* next_evaluation_points_out,
                                   __global int* is_integrator_still_running)
{
  int gid = get_global_id(0);

  if(gid < num_integrators)
  {
    scalar state = integration_state[gid];
    scalar position = current_position[gid];
    scalar stepsize = current_step_size[gid];
    scalar first_evaluation = range_begin_evaluation[gid];

    int is_still_running = 0;

    if(position < integration_end)
    {

      rkf_integrand_values evals = evaluations[gid];

      rkf_evaluation_points next_evaluation_points;
      rkf_advance(&state,
                  &position,
                  &stepsize,
                  &first_evaluation,
                  absolute_tolerance, relative_tolerance,
                  integration_end,
                  evals,
                  &next_evaluation_points,
                  EPSILON,
                  MINIMUM_STEPSIZE,
                  MAXIMUM_STEPSIZE);

      next_evaluation_points_out[gid] = next_evaluation_points;
      integration_state[gid] = state;
      current_position[gid] = position;
      current_step_size[gid] = stepsize;
      range_begin_evaluation[gid] = first_evaluation;
      is_still_running = 1;
    }

    is_integrator_still_running[gid] = is_still_running;
  }
}

__kernel void construct_evaluation_points_over_camera_plane(
                                       __global rkf_evaluation_points* integrator_required_evaluation_points,
                                       __global int* cumulative_num_running_integrators,
                                       __global int* is_integrator_still_running,
                                       vector4 camera_look_at,
                                       vector4 camera_x_basis,
                                       vector4 camera_y_basis,
                                       vector4 camera_plane_min_position,
                                       int num_pixels_x,
                                       int num_pixels_y,
                                       scalar pixel_size,
                                       __global vector4* evaluation_points_out)
{
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int integrator_id = gid_y * num_pixels_x + gid_x;

  if(gid_x < num_pixels_x && gid_y < num_pixels_y)
  {
    vector4 pixel_pos = camera_plane_min_position
                      + gid_x * pixel_size * camera_x_basis
                      + gid_y * pixel_size * camera_y_basis;

    rkf_evaluation_points required_z_values =
        integrator_required_evaluation_points[integrator_id];

    vector4 eval_points [4];

    for(int i = 0; i < 4; ++i)
      eval_points[i] = pixel_pos;

    eval_points[0] += required_z_values.s0 * camera_look_at;
    eval_points[1] += required_z_values.s1 * camera_look_at;
    eval_points[2] += required_z_values.s2 * camera_look_at;
    eval_points[3] += required_z_values.s3 * camera_look_at;

    if(is_integrator_still_running[integrator_id])
    {
      int evaluation_id = cumulative_num_running_integrators[integrator_id];

      for(int i = 0; i < 4; ++i)
        evaluation_points_out[4 * evaluation_id + i] = eval_points[i];
    }
  }
}

__kernel void gather_integrand_evaluations(__global scalar* reconstructor_results,
                                           __global int* cumulative_num_running_integrators,
                                           __global int* is_integrator_running,
                                           int num_integrators,
                                           __global rkf_integrand_values* evaluations_out)
{
  int gid = get_global_id(0);

  if(gid < num_integrators)
  {
    if(is_integrator_running[gid])
    {
      int evaluation_id = cumulative_num_running_integrators[gid];
      rkf_integrand_values evaluations;
      evaluations.s0 = reconstructor_results[4 * evaluation_id + 0];
      evaluations.s1 = reconstructor_results[4 * evaluation_id + 1];
      evaluations.s2 = reconstructor_results[4 * evaluation_id + 2];
      evaluations.s3 = reconstructor_results[4 * evaluation_id + 3];

      evaluations_out[gid] = evaluations;
    }
  }
}






#endif
