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


#ifndef VOLUMETRIC_NN8_RECONSTRUCTION
#define VOLUMETRIC_NN8_RECONSTRUCTION

#include "particle_grid_iteration_order.cl"
#include "particle_grid.cl"
#include "types.cl"

scalar distance2(vector2 a, vector2 b)
{
  vector2 R = a-b;
  return dot(R,R);
}

scalar distance23d(vector3 a, vector3 b)
{
  vector3 R = a-b;
  return dot(R,R);
}


typedef float8 nearest_neighbors_list;

inline
scalar get_weight3d(scalar distance)
{
  scalar dist2_inv = 1.f / (distance*distance);
  scalar dist4_inv = dist2_inv * dist2_inv;
  return dist4_inv * dist4_inv * dist2_inv;
}



#define MAX_CONTRIBUTION_DISTANCE MAXFLOAT


void nn_list_to_array8(nearest_neighbors_list nn_list,
                       float* out)
{
  out[0] = nn_list.s0;
  out[1] = nn_list.s1;
  out[2] = nn_list.s2;
  out[3] = nn_list.s3;
  out[4] = nn_list.s4;
  out[5] = nn_list.s5;
  out[6] = nn_list.s6;
  out[7] = nn_list.s7;
}

nearest_neighbors_list array8_to_nn_list(float* array)
{
  nearest_neighbors_list out;
  out.s0 = array[0];
  out.s1 = array[1];
  out.s2 = array[2];
  out.s3 = array[3];

  out.s4 = array[4];
  out.s5 = array[5];
  out.s6 = array[6];
  out.s7 = array[7];
  return out;
}

#define NUM_NEIGHBORS 8

__kernel void volumetric_nn8_reconstruction(
    int is_first_run,
    __global int2* grid_cells,
    int3 num_grid_cells,
    vector3 grid_min_corner,
    vector3 grid_cell_sizes,

    __global vector4* particles,

    int num_evaluation_points,
    __global vector4* evaluation_points_coordinates,
    __global nearest_neighbors_list* evaluation_points_weights,
    __global nearest_neighbors_list* evaluation_points_values)
{
  int gid = get_global_id(0);

  if (gid < num_evaluation_points)
  {
    grid3d_ctx grid;
    grid3d_init(&grid, grid_min_corner, grid_cell_sizes, num_grid_cells);

    vector3 evaluation_point_coord = evaluation_points_coordinates[gid].xyz;
    scalar distances [NUM_NEIGHBORS];
    scalar values    [NUM_NEIGHBORS];

    if (is_first_run)
    {
      for(int i = 0; i < NUM_NEIGHBORS; ++i)
      {
        distances[i] = MAX_CONTRIBUTION_DISTANCE;
        values[i]  = 0.0f;
      }
    }
    else
    {
      nearest_neighbors_list previous_weights = evaluation_points_weights[gid];
      nearest_neighbors_list previous_values = evaluation_points_values[gid];

      nn_list_to_array8(previous_weights, distances);
      nn_list_to_array8(previous_values, values);
    }

    int3 evaluation_point_cell = grid3d_get_cell_indices(&grid, evaluation_point_coord);
    vector3 evaluation_point_cell_min_corner = grid3d_get_cell_min_corner(&grid,
                                                                          evaluation_point_cell);

    scalar cell_radius = 0.5f * sqrt(dot(grid.cell_sizes, grid.cell_sizes));

    scalar search_radius = 0.0f;
    int max_distance_neighbor = 0;

    for(int i = 0; i < NUM_NEIGHBORS; ++i)
    {
      if(distances[i] > search_radius)
        max_distance_neighbor = i;
      search_radius = fmax(search_radius, distances[i]);
    }


    int subcell_id = 0;
    {
      grid3d_ctx subgrid;
      grid3d_init(&subgrid,
                  evaluation_point_cell_min_corner,
                  1.f/(float)NUM_SUBCELLS * grid_cell_sizes,
                  (int3)(NUM_SUBCELLS, NUM_SUBCELLS, NUM_SUBCELLS));

      subcell_id = grid3d_get_cell_key(&subgrid, evaluation_point_coord);
    }

    for(int i = 0; i < MAX_NUMBER_ITERATED_TILES; ++i)
    {
      char4 cell_delta =
        spiral3d_grid_iteration_order[subcell_id * MAX_NUMBER_ITERATED_TILES + i];

      int3 current_cell = evaluation_point_cell;
      current_cell += (int3)((int)cell_delta.x,
                             (int)cell_delta.y,
                             (int)cell_delta.z);

      vector3 current_cell_center = grid3d_get_cell_center(&grid, current_cell);

      scalar r = search_radius + cell_radius;
      if(distance23d(current_cell_center, evaluation_point_coord) > r*r)
        break;

      if(grid3d_is_cell_in_grid(&grid, current_cell))
      {
        ulong current_cell_key = grid3d_get_cell_key_from_indices(&grid, current_cell);
        int2 cell_data = grid_cells[current_cell_key];

        for(int current_particle_id = cell_data.x;
            current_particle_id < cell_data.y;
            ++current_particle_id)
        {
          vector4 current_particle = particles[current_particle_id];
          scalar current_distance = distance(current_particle.xyz, evaluation_point_coord);

          if(current_distance < search_radius)
          {
            // Overwrite the neighbor with the largest distance
            // with our newly found neighbor
            distances[max_distance_neighbor] = current_distance;
            values[max_distance_neighbor] = current_particle.w;

            // Update search radius
            search_radius = 0.0f;
            for(int j = 0; j < NUM_NEIGHBORS; ++j)
            {
              if(distances[j] > search_radius)
                max_distance_neighbor = j;
              search_radius = fmax(search_radius, distances[j]);
            }
          }
        }
      }
    }


    evaluation_points_weights[gid] = array8_to_nn_list(distances);
    evaluation_points_values [gid] = array8_to_nn_list(values);
  }

}

__kernel void finalize_volumetric_nn8_reconstruction(int num_evaluation_points,
                                                 __global nearest_neighbors_list* evaluation_points_weights,
                                                 __global nearest_neighbors_list* evaluation_points_values,
                                                 __global scalar* output)
{
  int gid = get_global_id(0);

  if(gid < num_evaluation_points)
  {

    nearest_neighbors_list distances = evaluation_points_weights[gid];
    nearest_neighbors_list weights;
    weights.s0 = get_weight3d(distances.s0);
    weights.s1 = get_weight3d(distances.s1);
    weights.s2 = get_weight3d(distances.s2);
    weights.s3 = get_weight3d(distances.s3);
    weights.s4 = get_weight3d(distances.s4);
    weights.s5 = get_weight3d(distances.s5);
    weights.s6 = get_weight3d(distances.s6);
    weights.s7 = get_weight3d(distances.s7);

    nearest_neighbors_list values  = evaluation_points_values [gid];

    scalar weight_sum = 0.0f;
    weight_sum += weights.s0;
    weight_sum += weights.s1;
    weight_sum += weights.s2;
    weight_sum += weights.s3;
    weight_sum += weights.s4;
    weight_sum += weights.s5;
    weight_sum += weights.s6;
    weight_sum += weights.s7;

    scalar dot_product = 0.0f;
    // Go FMA, baby!
    dot_product = fma(weights.s0, values.s0, dot_product);
    dot_product = fma(weights.s1, values.s1, dot_product);
    dot_product = fma(weights.s2, values.s2, dot_product);
    dot_product = fma(weights.s3, values.s3, dot_product);
    dot_product = fma(weights.s4, values.s4, dot_product);
    dot_product = fma(weights.s5, values.s5, dot_product);
    dot_product = fma(weights.s6, values.s6, dot_product);
    dot_product = fma(weights.s7, values.s7, dot_product);
    /*dot_product += weights.s0 * values.s0;
    dot_product += weights.s1 * values.s1;
    dot_product += weights.s2 * values.s2;
    dot_product += weights.s3 * values.s3;
    dot_product += weights.s4 * values.s4;
    dot_product += weights.s5 * values.s5;
    dot_product += weights.s6 * values.s6;
    dot_product += weights.s7 * values.s7;*/

    scalar result = 0.0f;
    if(weight_sum != 0.0f && dot_product != 0.0f)
      result = dot_product / weight_sum;

    output[gid] = result;
  }
}

#endif
