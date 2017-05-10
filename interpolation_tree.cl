

typedef float4 particle;
typedef int8 children_list;
typedef int particle_counter;
typedef float4 coordinate;
typedef float3 vector3;
typedef float scalar;

typedef int global_cell_id;
typedef char subcell_id;

#define MAX_TREE_DEPTH 16




scalar get_node_width(int level, scalar root_diameter)
{
  return root_diameter / (float)(1 << level);
}

int node_can_be_approximated(coordinate node_center,
                             scalar node_width,
                             coordinate evaluation_point,
                             scalar opening_angle_squared)
{
  vector3 R = node_center.xyz - evaluation_point.xyz;
  scalar r2 = dot(R,R);

  return node_width * node_width / r2 < opening_angle_squared;
}

int is_node_leaf(global_cell_id* subcells)
{
  int sum = 0;
  for(int i = 0; i < 8; ++i)
  {
    sum += subcells[i];
  }
  // Unused subcells are marked by a negative value
  return sum <= -8;
}

scalar get_weight(vector3 particle_point, vector3 evaluation_point)
{
  vector3 R = particle_point - evaluation_point;
  scalar r2_inv = 1.f / dot(R,R);
  return r2_inv*r2_inv*r2_inv;
}

void fetch_children(global_cell_id cell,
                    __global children_list* subcells,
                    global_cell_id* out)
{
  children_list children = subcells[cell];
  out[0] = children.s0;
  out[1] = children.s1;
  out[2] = children.s2;
  out[3] = children.s3;
  out[4] = children.s4;
  out[5] = children.s5;
  out[6] = children.s6;
  out[7] = children.s7;
}


void evaluate_cell(global_cell_id cell,
                   coordinate evaluation_point,
                   __global particle* particles,
                   __global particle_counter* num_particles,
                   __global coordinate* mean_coordinates,
                   scalar* result,
                   scalar* weight_sum)
{
  particle p = particles[cell];

  *result += p.w * get_weight(p.xyz, evaluation_point.xyz);

  *weight_sum += num_particles[cell]
                * get_weight(mean_coordinates[cell].xyz, evaluation_point.xyz);

}

/*
 * Recursive formulation of the following tree walk
 * algorithm (much easier to understand!):
 *
 * scalar interpolate(cell)
 * {
 *   if(cell is leaf || can be approximated)
 *      return cell evaluation
 *   else
 *   {
 *     scalar result;
 *     for(subcell : cell.subcells)
 *     {
 *       if(subcell is nonempty)
 *          result += interpolate(subcell)
 *     }
 *     return result;
 *   }
 * }
 *
 */

typedef struct
{
  global_cell_id current_cell;
  subcell_id next_subcell;
  global_cell_id children [8];
} stack_entry;

void interpolate(__global particle* particles,
                   __global children_list* subcells,
                   __global particle_counter* num_particles,
                   __global coordinate* mean_coordinates,
                   coordinate root_center,
                   scalar root_diameter,

                   coordinate evaluation_point,
                   scalar opening_angle,
                   scalar* value_sum_out,
                   scalar* weight_sum_out)
{
  scalar opening_angle2 = opening_angle * opening_angle;

  stack_entry stack [MAX_TREE_DEPTH];
  int stack_frame = 0;

  stack[0].current_cell = 0;
  stack[0].next_subcell = -1;
  fetch_children(0, subcells, stack[0].children);

  scalar current_result = 0.0f;
  scalar current_weight_sum = 0.0f;

  while(stack_frame >= 0)
  {

    if(stack[stack_frame].next_subcell >= 8)
    // The cell has already been fully processed - pop current stack frame
    {
      --stack_frame;
      continue;
    }

    scalar     cell_width = get_node_width(stack_frame, root_diameter);
    coordinate cell_center = particles[stack[stack_frame].current_cell];


    int is_leaf = is_node_leaf(stack[stack_frame].children);
    int is_far_away = node_can_be_approximated(cell_center,
                                               cell_width,
                                               evaluation_point,
                                               opening_angle2);

    if(is_leaf || is_far_away)
    {
      evaluate_cell(stack[stack_frame].current_cell,
                    evaluation_point,
                    particles,
                    num_particles,
                    mean_coordinates,
                    &current_result,
                    &current_weight_sum);

      // Pop stack frame
      --stack_frame;
    }
    else
    {
      // Now we need to investigate the subcells
      // First, find next unprocessed subcell
      int i = stack[stack_frame].next_subcell + 1;
      for(; i < 8; ++i)
        if(stack[stack_frame].children[i] >= 0)
          break;
      stack[stack_frame].next_subcell = i;

      if(i >= 8)
        // No more subcells to process - pop stack frame
        --stack_frame;
      else
      {
        // Push subcell onto stack
        ++stack_frame;
        stack[stack_frame].current_cell = stack[stack_frame - 1].children[i];
        stack[stack_frame].next_subcell = -1;
        fetch_children(stack[stack_frame - 1].children[i],
                       subcells,
                       stack[stack_frame].children);
      }
    }
  }

  *value_sum_out = current_result;
  *weight_sum_out = current_weight_sum;
}



__kernel void tree_interpolation(int is_first_run,
                                 __global particle* particles,
                                 __global children_list* subcells,
                                 __global particle_counter* num_particles,
                                 __global coordinate* mean_coordinates,
                                 coordinate root_center,
                                 scalar root_diameter,

                                 scalar opening_angle,

                                 __global coordinate* evaluation_points,
                                 int num_evaluation_points,
                                 __global scalar* value_sum_state,
                                 __global scalar* weight_sum_state,
                                 __global scalar* results)
{
  int gid = get_global_id(0);

  if(gid < num_evaluation_points)
  {
    coordinate evaluation_point = evaluation_points[gid];

    scalar value_sum;
    scalar weight_sum;

    interpolate(particles,
                subcells,
                num_particles,
                mean_coordinates,
                root_center,
                root_diameter,
                evaluation_point,
                opening_angle,
                &value_sum,
                &weight_sum);

    if(is_first_run)
    {
      value_sum_state [gid] = value_sum;
      weight_sum_state[gid] = weight_sum;
    }
    else
    {
      value_sum_state [gid] += value_sum;
      weight_sum_state[gid] += weight_sum;
    }


    results[gid] = value_sum / weight_sum;
  }
}

