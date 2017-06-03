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

#include "interpolation.cl"
#include "volumetric_nn8_tile_iteration_order.cl"



typedef float8 nearest_neighbors_list;

__constant uint16 insertion_shuffle_masks [] =
{
  (uint16)(8, 0, 1, 2, 3, 4, 5, 6, (uint8)(8)),
  (uint16)(0, 8, 1, 2, 3, 4, 5, 6, (uint8)(8)),
  (uint16)(0, 1, 8, 2, 3, 4, 5, 6, (uint8)(8)),
  (uint16)(0, 1, 2, 8, 3, 4, 5, 6, (uint8)(8)),
  (uint16)(0, 1, 2, 3, 8, 4, 5, 6, (uint8)(8)),
  (uint16)(0, 1, 2, 3, 4, 8, 5, 6, (uint8)(8)),
  (uint16)(0, 1, 2, 3, 4, 5, 8, 6, (uint8)(8)),
  (uint16)(0, 1, 2, 3, 4, 5, 6, 8, (uint8)(8)),
  (uint16)(0, 1, 2, 3, 4, 5, 6, 7, (uint8)(8))
};



int3 get_tile_around_pos3(vector3 coordinate,
                      vector3 tiles_min_corner,
                      vector3 tile_size)
{
  vector3 float_result = (coordinate - tiles_min_corner) / tile_size;

  return (int3)((int)float_result.x, (int)float_result.y, (int)float_result.z);
}

void neighbor_list_insert(nearest_neighbors_list* list,
                   scalar value,
                   int insertion_pos)
{
  nearest_neighbors_list insertion_vector = (nearest_neighbors_list)value;
  *list = shuffle2(*list, insertion_vector, insertion_shuffle_masks[insertion_pos]).lo;
}

inline
void sorted_neighbors_insert(nearest_neighbors_list* distances,
                             nearest_neighbors_list* values,
                             scalar new_distance,
                             scalar new_value)
{
  // Find position to insert before
  int insertion_pos = 0;

  if(distances->s1 >= new_distance && new_distance > distances->s0)
    insertion_pos = 1;
  if(distances->s2 >= new_distance && new_distance > distances->s1)
    insertion_pos = 2;
  if(distances->s3 >= new_distance && new_distance > distances->s2)
    insertion_pos = 3;
  if(distances->s4 >= new_distance && new_distance > distances->s3)
    insertion_pos = 4;
  if(distances->s5 >= new_distance && new_distance > distances->s4)
    insertion_pos = 5;
  if(distances->s6 >= new_distance && new_distance > distances->s5)
    insertion_pos = 6;
  if(distances->s7 >= new_distance && new_distance > distances->s6)
    insertion_pos = 7;
  if(new_distance > distances->s7)
    insertion_pos = 8;

  neighbor_list_insert(distances, new_distance, insertion_pos);
  neighbor_list_insert(values, new_value, insertion_pos);
}

inline
scalar nearest_neighbor_list_max(nearest_neighbors_list x)
{
  return x.s7;
}

inline
scalar nearest_neighbor_list_min(nearest_neighbors_list x)
{
  return x.s0;
}

inline
scalar get_weight3d(scalar distance2)
{
  scalar dist2_inv = 1.f / distance2;
  scalar dist4_inv = dist2_inv * dist2_inv;
  return dist4_inv * dist4_inv * dist2_inv;
}

inline
int is_tile_id_component_valid(int id, int num_tiles)
{
  return id >= 0 && id < num_tiles;
}

inline
int tile_exists(int3 tile_id, int3 num_tiles)
{
  if(!is_tile_id_component_valid(tile_id.x, num_tiles.x))
    return false;
  if(!is_tile_id_component_valid(tile_id.y, num_tiles.y))
    return false;
  if(!is_tile_id_component_valid(tile_id.z, num_tiles.z))
    return false;
  return true;
}

#define MAX_CONTRIBUTION_DISTANCE MAXFLOAT

inline
void evaluate_tile(int3 current_tile,
                   __global vector4* tiles,
                   int3 num_tiles,
                   __global vector4* particles,
                   vector3 evaluation_point_coord,
                   nearest_neighbors_list* distances,
                   nearest_neighbors_list* values,
                   scalar* search_radius)
{
  vector4 tile_header =
      tiles[current_tile.z * num_tiles.x * num_tiles.y +
            current_tile.y * num_tiles.x +
            current_tile.x];

  int num_particles_in_tile = (int)tile_header.x;

  // scalar maximum_smoothing_distance_of_tile = tile_header.y;
  int particle_data_offset = (int)tile_header.z;

  //if(get_global_id(0)==512)
  //  printf("n_particles = %d\n", num_particles_in_tile);

  for (int i = 0; i < num_particles_in_tile; ++i)
  {
    vector4 current_particle = particles[particle_data_offset + i];

    scalar particle_distance2 =
        distance23d(current_particle.xyz, evaluation_point_coord);

    if (particle_distance2 < nearest_neighbor_list_max(*distances))
    {

      scalar new_value = current_particle.w;

      sorted_neighbors_insert(distances, values, particle_distance2,
                                new_value);

      // If we already have all slots for weights taken, we can
      // assume as search radius the distance to the particle with
      // the lowest contribution (i.e. the farthest particle)
      *search_radius = fmin(*search_radius, sqrt(nearest_neighbor_list_max(*distances)));

    }
  }
}

__kernel void volumetric_nn8_reconstruction(
    int is_first_run,
    __global vector4* tiles,
    int3 num_tiles,
    vector3 tiles_min_corner,
    vector3 tile_sizes,

    __global vector4* particles,
    scalar maximum_smoothing_length,

    int num_evaluation_points,
    __global vector4* evaluation_points_coordinates,
    __global nearest_neighbors_list* evaluation_points_weights,
    __global nearest_neighbors_list* evaluation_points_values)
{
  int gid = get_global_id(0);

  if (gid < num_evaluation_points)
  {
    vector3 evaluation_point_coord = evaluation_points_coordinates[gid].xyz;
    nearest_neighbors_list distances;
    nearest_neighbors_list values;
    if (is_first_run)
    {
      distances = (nearest_neighbors_list)(MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE,
                                           MAX_CONTRIBUTION_DISTANCE);
      values  = (nearest_neighbors_list)(0, 0, 0, 0, 0, 0, 0, 0);
    }
    else
    {
      distances = evaluation_points_weights[gid];
      values =  evaluation_points_values[gid];
    }


    int3 evaluation_tile = get_tile_around_pos3(evaluation_point_coord,
                                                tiles_min_corner, tile_sizes);

    vector3 tile_min_corner =
        tiles_min_corner + (vector3)((float)evaluation_tile.x,
                                     (float)evaluation_tile.y,
                                     (float)evaluation_tile.z) * tile_sizes;
    vector3 tile_center =
        tile_min_corner + 0.5f * tile_sizes;

    scalar tile_radius = 0.5f * sqrt(dot(tile_sizes, tile_sizes));


    scalar search_radius = maximum_smoothing_length;
    if(tile_exists(evaluation_tile, num_tiles))
    {
      vector4 evaluation_tile_header =
        tiles[evaluation_tile.z * num_tiles.x * num_tiles.y +
              evaluation_tile.y * num_tiles.x +
              evaluation_tile.x];

      search_radius = evaluation_tile_header.w;
    }

    if(search_radius == 0.0f)
      search_radius = maximum_smoothing_length;

    search_radius = fmin(search_radius, nearest_neighbor_list_max(distances));


    int3 subtile = get_tile_around_pos3(evaluation_point_coord,
                                        tile_min_corner, 1.f/(float)NUM_SUBTILES * tile_sizes);
    int subtile_id = subtile.x
                   + subtile.y * NUM_SUBTILES
                   + subtile.z * NUM_SUBTILES * NUM_SUBTILES;

    for(int i = 0; i < MAX_NUMBER_ITERATED_TILES; ++i)
    {
      char4 tile_delta =
        spiral3d_grid_iteration_order[subtile_id * MAX_NUMBER_ITERATED_TILES + i];

      int3 current_tile = evaluation_tile;
      current_tile.x += tile_delta.x;
      current_tile.y += tile_delta.y;
      current_tile.z += tile_delta.z;

      vector3 current_tile_center = tiles_min_corner;
      current_tile_center.x += ((float)current_tile.x + 0.5f) * tile_sizes.x;
      current_tile_center.y += ((float)current_tile.y + 0.5f) * tile_sizes.y;
      current_tile_center.z += ((float)current_tile.z + 0.5f) * tile_sizes.z;

      scalar r = search_radius+tile_radius;
      if(distance23d(current_tile_center, evaluation_point_coord) > r*r)
        break;

      if(tile_exists(current_tile, num_tiles))
      {
        evaluate_tile(current_tile,
                      tiles, num_tiles, particles,
                      evaluation_point_coord,
                      &distances, &values,
                      &search_radius);
      }
    }


    evaluation_points_weights[gid] = distances;
    evaluation_points_values[gid] = values;
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
    dot_product += weights.s0 * values.s0;
    dot_product += weights.s1 * values.s1;
    dot_product += weights.s2 * values.s2;
    dot_product += weights.s3 * values.s3;
    dot_product += weights.s4 * values.s4;
    dot_product += weights.s5 * values.s5;
    dot_product += weights.s6 * values.s6;
    dot_product += weights.s7 * values.s7;

    scalar result = 0.0f;
    if(weight_sum != 0.0f && dot_product != 0.0f)
      result = dot_product / weight_sum;

    output[gid] = result;
  }
}

#endif
