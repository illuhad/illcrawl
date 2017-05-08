

typedef float4 particle;
typedef int8 children_list;
typedef int particle_counter;
typedef float3 coordinate;
typedef float scalar;

typedef int global_cell_id;
typedef int subcell_id;

#define MAX_TREE_DEPTH 16;

typedef struct stack_entry
{
  global_cell_id current_cell;
  subcell_id next_subcell;
};

__constant coordinate subcell_directional_sign [8] =
{
   (coordinate)(-1, -1, -1), // subcell_id = 0
   (coordinate)( 1, -1, -1), // subcell_id = 1
   (coordinate)(-1,  1, -1), // subcell_id = 2
   (coordinate)( 1,  1, -1), // subcell_id = 3
   (coordinate)(-1, -1,  1), // subcell_id = 4
   (coordinate)( 1, -1,  1), // subcell_id = 5
   (coordinate)(-1,  1,  1), // subcell_id = 6
   (coordinate)( 1,  1,  1)  // subcell_id = 7
};

scalar get_node_width(int level, scalar root_diameter)
{
  return root_diameter / (float)(1 << level);
}

coordinate get_node_center(coordinate parent_center,
                           scalar parent_width,
                           subcell_id id)
{
  return parent_center
       + subcell_directional_sign[id] * (coordinate)(0.25f * parent_width);
}

int node_can_be_approximated(coordinate node_center,
                             scalar node_width,
                             coordinate evaluation_point,
                             scalar opening_angle_squared)
{
  coordinate R = node_center - evaluation_point;
  scalar r2 = dot(R,R);

  return node_width * node_width / r2 < opening_angle_squared;
}

int is_node_leaf(children_list subcells)
{
  int sum = 0;
  sum += subcells.s0;
  sum += subcells.s1;
  sum += subcells.s2;
  sum += subcells.s3;
  sum += subcells.s4;
  sum += subcells.s5;
  sum += subcells.s6;
  sum += subcells.s7;

  // Unused subcells are marked by a negative value
  return sum <= -8;
}

scalar get_weight(coordinate particle_point, coordinate evaluation_point)
{
  coordinate R = particle_point - evaluation_point;
  return 1.f / dot(R,R);
}

scalar process_cell(stack_entry* stack,
                    int stack_frame)
{

}

scalar interpolate(__global particle* particles,
                   __global children_list* subcells,
                   __global particle_counter* num_particles,
                   __global coordinate* mean_coordinates,
                   coordinate root_center,
                   scalar root_diameter,

                   coordinate evaluation_point,
                   scalar opening_angle)
{

  stack_entry stack [MAX_TREE_DEPTH];
  int stack_frame = 0;

  stack[0].current_cell = 0;
  stack[0].next_subcell = 0;

  scalar opening_angle2 = opening_angle * opening_angle;

  scalar current_result = 0.0f;
  scalar current_weight_sum = 0.0f;

  while(stack[0].next_subcell < 8)
  {
    coordinate cell_center = //ToDo;
    scalar     cell_width = get_node_width(stack_frame, root_diameter);

    global_cell_id cell = stack[stack_frame].current_cell;
    children_list cell_subcells = subcells[cell];
    int is_leaf = is_node_leaf(cell_subcells);

    if(is_leaf || node_can_be_approximated(cell_center,
                                          cell_width,
                                          evaluation_point,
                                          opening_angle2))
    {
      particle p = particles[cell];

      current_result += p.w
                     * get_weight(p.xyz, evaluation_point);

      current_weight_sum += num_particles[cell]
                         * get_weight(mean_coordinates[cell], evaluation_point);


      stack[stack_frame-1].next_subcell += 1;
      if(stack[stack_frame-1].next_subcell > 7)
        // All nodes on this level have been processed - go to parent
        --stack_frame;
      else
      {
        // Go to the next sibling of the node
        stack[stack_frame].current_cell = stack[stack_frame-1].next_subcell;
      }
    }
    else
    {
      // We must investigate the subnodes
      stack[stack_frame + 1].current_cell = cell_subcells[0]
      stack[stack_frame + 1].next_subcell = 0;

      ++stack_frame;
    }
  }

  return current_result / current_weight_sum;
}

__kernel void tree_interpolation(__global particle* particles,
                                 __global children_list* subcells,
                                 __global particle_counter* num_particles,
                                 __global coordinate* mean_coordinates,
                                 coordinate root_center,
                                 scalar root_diameter,

                                 scalar opening_angle,

                                 __global coordinate* evaluation_points,
                                 int num_evaluation_points,
                                 __global scalar* results)
{
  int gid = get_global_id(0);

  if(gid < num_evaluation_points)
  {
    coordinate evaluation_point = evaluation_points[gid];

    scalar result = interpolate(particles,
                                subcells,
                                num_particles,
                                mean_coordinates,
                                root_center,
                                root_diameter,
                                evaluation_point,
                                opening_angle);
    results[gid] = result;
  }
}

