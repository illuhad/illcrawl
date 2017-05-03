#ifndef RECONSTRUCTION_CL
#define RECONSTRUCTION_CL

#include "interpolation.cl"



int2 get_tile_around_pos(vector2 coordinate,
                      vector2 tiles_min_corner,
                      vector2 tile_size)
{
  vector2 float_result = (coordinate - tiles_min_corner) / tile_size;

  return (int2)((int)float_result.x, (int)float_result.y);
}

// 3d version
int3 get_tile_around_pos3(vector3 coordinate,
                      vector3 tiles_min_corner,
                      vector3 tile_size)
{
  vector3 float_result = (coordinate - tiles_min_corner) / tile_size;

  return (int3)((int)float_result.x, (int)float_result.y, (int)float_result.z);
}

// CLK_ADDRESS_CLAMP will return (vector4)(0.0f) outside the bounds of
// the image for CL_RGBA images.
//const sampler_t tile_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;

__kernel void image_tile_based_reconstruction2D(__global vector4* tiles,
                                                __global vector4* particles,
                                                scalar maximum_smoothing_length,
                                                __global scalar* pixels_out,
                                                vector2 pixels_min_corner,
                                                vector2 pixels_max_corner,
                                                unsigned num_px_x,
                                                unsigned num_px_y,
                                                unsigned num_tiles_x,
                                                unsigned num_tiles_y,
                                                //__local vector4* local_mem,
                                                //__local scalar* smoothing_length_cache,
                                                __global scalar* quantity)
{
  int tid_x = get_global_id(0);
  int tid_y = get_global_id(1);

  //int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
  //int total_local_size = get_local_size(0)*get_local_size(1);
  //int depth = get_image_depth(tiles);

  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);

  scalar dx = (pixels_max_corner.x - pixels_min_corner.x) / (float)num_px_x;
  scalar dy = (pixels_max_corner.y - pixels_min_corner.y) / (float)num_px_y;

  vector2 tile_min = pixels_min_corner;
  tile_min.x += get_group_id(0) * get_local_size(0) * dx;
  tile_min.y += get_group_id(1) * get_local_size(1) * dy;

  vector2 tile_size;
  tile_size.x = get_local_size(0) * dx;
  tile_size.y = get_local_size(1) * dy;

  vector2 tile_max = tile_min;
  tile_max.x += tile_size.x * dx;
  tile_max.y += tile_size.y * dy;

  vector2 tile_center = 0.5f * (tile_min + tile_max);
  scalar tile_radius = 0.5f * sqrt(tile_size.x*tile_size.x + tile_size.y*tile_size.y);

  vector2 pixel_min = tile_min;
  pixel_min.x += get_local_id(0) * dx;
  pixel_min.y += get_local_id(1) * dy;

  vector2 pixel_max = pixel_min;
  pixel_max.x += dx;
  pixel_max.y += dy;

  vector2 px_center = 0.5f * (pixel_min + pixel_max);

  int2 min_tile = get_tile_around_pos(tile_center - (maximum_smoothing_length + tile_radius),
                                      pixels_min_corner,
                                      tile_size);
  min_tile.x = max(0, min_tile.x);
  min_tile.y = max(0, min_tile.y);

  int2 max_tile = get_tile_around_pos(tile_center + (maximum_smoothing_length + tile_radius),
                                      pixels_min_corner,
                                      tile_size);

  max_tile.x = min((int)(num_tiles_x-1), max_tile.x);
  max_tile.y = min((int)(num_tiles_y-1), max_tile.y);


  //if(get_global_id(0)==512 && get_global_id(1)==512)
  //  printf("center %f %f tile_min %f %f min_tile %d %d max_tile %d %d\n",tile_center.x,tile_center.y,tile_min.x,tile_min.y,min_tile.x,min_tile.y,max_tile.x,max_tile.y);
  scalar result = 0.0f;
  for(int current_tile_x = min_tile.x; current_tile_x <= max_tile.x; ++current_tile_x)
  {
    for(int current_tile_y = min_tile.y; current_tile_y <= max_tile.y; ++current_tile_y)
    {
      // Retrieve tile header
      int2 tile_coord = (int2)(current_tile_x, current_tile_y);
      vector4 header = tiles[tile_coord.y * num_tiles_x + tile_coord.x];

      int num_particles_in_tile = (int)header.x;
      scalar maximum_smoothing_distance_of_tile = header.y;
      int particle_data_offset = (int)header.z;

      scalar r = maximum_smoothing_distance_of_tile + tile_radius;
      vector2 current_tile_center = pixels_min_corner;
      current_tile_center.x += (current_tile_x + 0.5f) * tile_size.x;
      current_tile_center.y += (current_tile_y + 0.5f) * tile_size.y;

      if (distance2(current_tile_center, px_center) < r * r)
      {

        //for (int i = 0; i < num_particles_in_tile; i += total_local_size)
        for (int i = 0; i < num_particles_in_tile; ++i)
        {
          //if(i+local_id < num_particles_in_tile)
          //  local_mem[local_id] = particles[particle_data_offset + i + local_id];
          //barrier(CLK_LOCAL_MEM_FENCE);

          //for(int j = 0; i+j < num_particles_in_tile; ++j)
          //{
          //  vector4 particle = local_mem[j];
          vector4 particle = particles[particle_data_offset + i];
          scalar smoothing_length = particle.z;
          int particle_id = (int)particle.w;

          scalar weight = get_weight(particle.xy, smoothing_length, px_center,
                                     dx, dy, pixel_min, pixel_max);

          // if(get_global_id(0)==512 && get_global_id(1)==512)
          //  printf("%f %f %f %d %d\n", weight, particle.x, particle.y,
          //  current_tile_x, current_tile_y);
          if (weight != 0.0f)
            result += weight * quantity[particle_id];

        }
      }
    }
  }

  pixels_out[tid_y * num_px_x + tid_x] += result;
}

typedef float8 nearest_neighbors_list;

__constant uint8 insertion_shuffle_masks [] =
{
  (uint16)(0, 1, 2, 3, 4, 5, 6, 7, (uint8)(8)),
  (uint16)(8, 1, 2, 3, 4, 5, 6, 7, (uint8)(8)),
  (uint16)(1, 8, 2, 3, 4, 5, 6, 7, (uint8)(8)),
  (uint16)(1, 2, 8, 3, 4, 5, 6, 7, (uint8)(8)),
  (uint16)(1, 2, 3, 8, 4, 5, 6, 7, (uint8)(8)),
  (uint16)(1, 2, 3, 4, 8, 5, 6, 7, (uint8)(8)),
  (uint16)(1, 2, 3, 4, 5, 8, 6, 7, (uint8)(8)),
  (uint16)(1, 2, 3, 4, 5, 6, 8, 7, (uint8)(8)),
  (uint16)(1, 2, 3, 4, 5, 6, 7, 8, (uint8)(8))
};

void neighbor_list_insert(nearest_neighbors_list* list,
                   scalar value,
                   int insertion_pos)
{
  nearest_neighbors_list insertion_vector = (nearest_neighbors_list)value;
  *list = shuffle2(*list, insertion_vector, insertion_shuffle_masks[insertion_pos]).lo;
}

void sorted_neighbors_insert(nearest_neighbors_list* weights,
                             nearest_neighbors_list* values,
                             scalar new_weight,
                             scalar new_value)
{
  // Find position to insert before
  int insertion_pos = 0;
  for(; insertion_pos < 8; ++insertion_pos)
  {
    if(weights[insertion_pos] > new_weight)
      break;
  }

  neighbor_list_insert(weights, new_weight, insertion_pos);
  neighbor_list_insert(values, new_value, insertion_pos);
}

scalar nearest_neighbor_list_max(nearest_neighbors_list x)
{
  return x.s7;
}

scalar nearest_neighbor_list_min(nearest_neighbors_list x)
{
  return x.s0;
}

int is_nearest_neighbor_list_nonzero(nearest_neighbors_list x)
{
  // Due to the sorting, it is enough to check x.s7
  return x.s7 != 0.0f;
}


scalar get_weight3d(vector3 a, vector3 b)
{
  return 1.f / distance23d(a,b);
}

scalar get_distance_from_weight3d(scalar weight)
{
  return 1.f / sqrt(weight);
}


__kernel void reconstruction3D_weight_estimation(__global vector4* tiles,
                                                int3 num_tiles,
                                                __global vector4* particles,
                                                scalar maximum_smoothing_length,
                                                __global vector4* evaluation_points_coordinates,
                                                __global nearest_neighbors_list* evaluation_points_weights,
                                                __global nearest_neighbors_list* evaluation_points_values,
                                                vector3 tiles_min_corner,
                                                vector3 tile_sizes,
                                                __global scalar* quantity)
{
  int gid = get_global_id(0);

  vector3 evaluation_point_coord = evaluation_points_coordinates[gid].xyz;
  nearest_neighbors_list weights = evaluation_points_weights[gid];
  nearest_neighbors_list values = evaluation_points_values[gid];
  scalar maximum_weight = nearest_neighbor_list_max(weights);

  int3 evaluation_tile = get_tile_around_pos3(evaluation_point_coord, tiles_min_corner, tile_sizes);

  vector3 tile_center = tiles_min_corner + (evaluation_tile + 0.5f) * tile_sizes;
  scalar tile_radius = 0.5f * sqrt(dot(tile_sizes, tile_sizes));

  scalar search_radius = maximum_smoothing_length;
  if(is_nearest_neighbor_list_nonzero(weights))
    search_radius = get_distance_from_weight(nearest_neighbor_list_min(weights));

  int3 min_tile = get_tile_around_pos3(tile_center - (search_radius + tile_radius),
                                      tiles_min_corner,
                                      tile_size);
  min_tile.x = max(0, min_tile.x);
  min_tile.y = max(0, min_tile.y);
  min_tile.z = max(0, min_tile.z);

  int3 max_tile = get_tile_around_pos3(tile_center + (search_radius + tile_radius),
                                      tiles_min_corner,
                                      tile_size);

  max_tile.x = min((int)(num_tiles.x-1), max_tile.x);
  max_tile.y = min((int)(num_tiles.y-1), max_tile.y);
  max_tile.z = min((int)(num_tiles.z-1), max_tile.z);

  int3 current_tile;
  for(current_tile.z = min_tile.z;
      current_tile.z <= max_tile.z;
      ++current_tile.z)
  {
    for(current_tile.y = min_tile.y;
        current_tile.y <= max_tile.y;
        ++current_tile.y)
    {
      for(current_tile.x = min_tile.x;
          current_tile.x <= max_tile.x;
          ++current_tile.x)
      {
        vector3 current_tile_center = tiles_min_corner + (current_tile + 0.5f) * tile_sizes;
        scalar r = tile_radius + search_radius;

        if(distance23d(current_tile_center, evaluation_point_coord) < r*r)
        {

          vector4 tile_header = tiles[current_tile.z * num_tiles.x * num_tiles.y
                                    + current_tile.y * num_tiles.x
                                    + current_tile.x];

          int num_particles_in_tile = (int)tile_header.x;
          //scalar maximum_smoothing_distance_of_tile = tile_header.y;
          int particle_data_offset = (int)tile_header.z;

          scalar current_min_weight = nearest_neighbor_list_min(weights);

          for(int i = 0; i < num_particles_in_tile; ++i)
          {
            vector4 current_particle = particles[particle_data_offset + i];

            scalar new_weight = get_weight3d(current_particle.xyz, evaluation_point_coord);
            if(new_weight > current_min_weight)
            {
              scalar new_value = quantity[(int)current_particle.w];

              sorted_neighbors_insert(&weights, &values, new_weight, new_value);

              current_min_weight = nearest_neighbors_list_min(weights);

              // If we already have all slots for weights taken, we can
              // assume as search radius the distance to the particle with
              // the lowest contribution (i.e. the farthest particle)
              if(is_nearest_neighbor_list_nonzero(weights))
                search_radius = get_distance_from_weight(nearest_neighbor_list_min(weights));
            }
          }

        }
      }
    }
  }

  evaluation_points_weights[gid] = weights;
  evaluation_points_values[gid] = values;
}

__kernel void reconstruction3D(__global nearest_neighbors_list* evaluation_points_weights,
                               __global nearest_neighbors_list* evaluation_points_values,
                               __global scalar*                 evaluation_points_dz,
                               __global unsigned*               num_evaluation_points,
                               __global int*                    evaluation_point_offsets,
                               __global scalar* pixels_out,
                               unsigned num_pixels_x,
                               unsigned num_pixels_y,
                               scalar dx,
                               scalar dy)
{
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  scalar result = 0.0f;
  scalar dA = dx*dy;

  if(gid_x < num_pixels_x && gid_y < num_pixels_y)
  {
    int pos = gid_y * num_pixels_x + gid_x;
    unsigned num_evaluations = num_evaluation_points[pos];
    int offset               = evaluation_point_offsets[pos];

    for(int i = 0; i < num_evaluations; ++i)
    {
      nearest_neighbors_list weights = evaluation_points_weights[offset + i];
      nearest_neighbors_list values  = evaluation_points_values [offset + i];
      scalar dz                      = evaluation_points_dz     [offset + i];

      scalar weight_sum = 0.0f;
      weight_sum += weights.s0;
      weight_sum += weights.s1;
      weight_sum += weights.s2;
      weight_sum += weights.s3;
      weight_sum += weights.s4;
      weight_sum += weights.s5;
      weight_sum += weights.s6;
      weight_sum += weights.s7;
      if(weight_sum == 0.0f)
        weight_sum = 1.0f;

      result += dz * dot(weights, values) / weight_sum;
    }

    pixels_out[pos] = dA * result;
  }

}



#endif
