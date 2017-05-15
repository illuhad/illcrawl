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

typedef float scalar;
typedef float4 integrand_values;
typedef float4 evaluation_points;

#define EPSILON 0.01f;
#define MINIMUM_STEPSIZE 0.2f;




void rkf_advance(scalar* integration_state,
                 scalar* current_position,
                 scalar* current_step_size,
                 scalar* range_begin_evaluation,
                 scalar tolerance,
                 int is_relative_tolerance,
                 scalar integration_end,
                 integrand_values evaluations,
                 evaluation_points* next_evaluation_points)
{
  scalar delta4 =
       + 25.f/216.f    * *range_begin_evaluation
       + 1408.f/2565.f * evaluations.s0
       + 2197.f/4101.f * evaluations.s1
       - 1.f/5.f       * evaluations.s2;

  scalar delta5 =
       + 16.f/135.f      * *range_begin_evaluation
       + 6656.f/12825.f  * evaluations.s0
       + 28561.f/56430.f * evaluations.s1
       - 9.f/50.f        * evaluations.s2
       + 2.f/55.f        * evaluations.s3;

  delta4 *= *current_step_size;
  delta5 *= *current_step_size;

  scalar estimate4 = *integration_state + delta4;
  scalar estimate5 = *integration_state + delta5;

  *current_position += *current_step_size;

  scalar s = 2.0f;
  if(estimate4 != estimate5)
  {
    scalar error = fabs(estimate5 - estimate4);
    scalar absolute_tolerance = tolerance;

    if(is_relative_tolerance)
      absolute_tolerance = *integration_state / (*_current_position) * tolerance;

    s = pow(absolute_tolerance * (*current_step_size) / (2 * error), 0.25f);
  }

  scalar new_step_size = s * (*current_step_size);

  if(new_step_size < MINIMUM_STEPSIZE)
  {
    new_step_size = minimum_stepsize;
    s = new_step_size / (*current_step_size);
  }

  if(s < 0.95f)
  {
    // Reject approximation, go back to old position
    *current_position -= *current_step_size;
  }
  else
  {
    // Accept approximation
    *integration_state = estimate4;
    *range_begin_evaluation = evaluations.s2;
  }

  *current_step_size = new_step_size;

 if(*current_position + (*current_step_size) > integration_end)
    // The epsilon's job is to make sure that the condition
    // get_position() < integration range turns false and a integration loop
    // does not turn into an infinite loop.
    *current_step_size = integration_end - *current_step_size + EPSILON;

  next_evaluation_points->s0 = *current_position + 3.f/8.f * (*current_step_size);
  next_evaluation_points->s1 = *current_position + 12.f/13.f * (*current_step_size);
  next_evaluation_points->s2 = *current_position + (*current_step_size);
  next_evaluation_points->s2 = *current_position + 0.5f * (*current_step_size);
}



__kernel void runge_kutta_fehlberg(__global scalar* integration_state,
                                   __global scalar* current_position,
                                   __global scalar* current_step_size,
                                   __global scalar* evaluations,
                                   __global scalar* range_begin_evaluation,
                                   int num_integrators,
                                   scalar tolerance,
                                   int is_relative_tolerance,
                                   scalar integration_end,
                                   __global evaluation_points* next_evaluation_points_out)
{
  int gid = get_global_id(0)

  if(gid < num_integrators)
  {
    scalar state = integration_state[gid];
    scalar position = current_position[gid];
    scalar stepsize = current_stepsize[gid];
    scalar first_evaluation = range_begin_evaluation[gid];

    if(position < integration_end)
    {
      // Build integrand_values vector - yes, this memory
      // access pattern is not optimal...
      integrand_values evaluations;
      evaluations.s0 = evaluations[4 * gid + 0];
      evaluations.s1 = evaluations[4 * gid + 1];
      evaluations.s2 = evaluations[4 * gid + 2];
      evaluations.s3 = evaluations[4 * gid + 3];

      evaluation_points next_evaluation_points;
      rkf_advance(&state,
                  &position,
                  &stepsize,
                  &first_evaluation,
                  tolerance, is_relative_tolerance,
                  integration_end,
                  &next_evaluation_points);

      next_evaluation_points_out[gid] = next_evaluation_points;
      integration_state[gid] = state;
      current_position[gid] = position;
      current_stepsize[gid] = stepsize;
      range_begin_evaluation[gid] = first_evaluation;
    }
  }
}

__kernel void generate_required_evaluations_list(__global scalar* positions,
                                                 __global scalar* evaluations,
                                                 scalar integration_end
                                                 __local int* integrator_counter,
                                                 __global int* partial_sums,
                                                 __global int* integrator_ids)
{
  int gid = get_global_id(0);
  int lid = get_local_id(0);

  integrator_counter[lid] = 0;

  if(positions[gid] < integration_end)
  {
    // Include integrator in evaluation list
    integrator_counter[lid] = 1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int i = 0; i < lid; ++i)
  {
    int current_state = integrator_counter[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i+1 < get_local_size(0))
      integrator_counter[lid + 1] += current_state;

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}





#endif
