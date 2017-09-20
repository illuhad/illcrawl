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

#include "types.cl"
#include "math.cl"

void create_projection_matrix(matrix3x3_t* m,
                              vector4 projection_plane_basis0,
                              vector4 projection_plane_basis1,
                              vector4 projection_plane_normal)
{
  M00(m) = projection_plane_basis0.x;
  M10(m) = projection_plane_basis0.y;
  M20(m) = projection_plane_basis0.z;

  M01(m) = projection_plane_basis1.x;
  M11(m) = projection_plane_basis1.y;
  M21(m) = projection_plane_basis1.z;

  M02(m) = projection_plane_normal.x;
  M12(m) = projection_plane_normal.y;
  M22(m) = projection_plane_normal.z;

  matrix3x3_invert(m);
}

vector4 project(matrix3x3_t* projection_matrix,
                vector4 projection_center,
                vector4 x)
{
  vector4 result;
  result.xyz = matrix3x3_vector_mult(projection_matrix,
                                     x.xyz - projection_center.xyz);
  result.w = x.w;

  return result;
}

__kernel void project_particles(__global vector4* particles,
                                unsigned long num_particles,
                                scalar max_projection_distance,
                                vector4 camera_plane_center,
                                vector4 camera_plane_basis0,
                                vector4 camera_plane_basis1,
                                vector4 camera_plane_normal)
{
  matrix3x3_t projection_matrix;
  create_projection_matrix(&projection_matrix,
                           camera_plane_basis0,
                           camera_plane_basis1,
                           camera_plane_normal);

  size_t tid = get_global_id(0);

  if(tid < num_particles)
  {
    vector4 current_particle = particles[tid];

    current_particle = project(&projection_matrix,
                               camera_plane_center,
                               current_particle);

    if(current_particle.z < 0.0f ||
       current_particle.z > max_projection_distance)
    {
       // Particle is either behind the camera or
       // too far away (beyond the integration range).
       // We hence mask this particle out by setting
       // its mass to 0
       current_particle.w = 0.0f;
    }
    // Also mask current_particle.z to make sure they
    // are all treated as lying in one plane
    current_particle.z = 0.0f;

    particles[tid] = current_particle;
  }
}

__kernel void project_evaluation_points(__global vector4* evaluation_points,
                                        unsigned long num_particles,
                                        vector4 camera_plane_center,
                                        vector4 camera_plane_basis0,
                                        vector4 camera_plane_basis1,
                                        vector4 camera_plane_normal)
{
  matrix3x3_t projection_matrix;
  create_projection_matrix(&projection_matrix,
                           camera_plane_basis0,
                           camera_plane_basis1,
                           camera_plane_normal);

  size_t tid = get_global_id(0);

  if(tid < num_particles)
  {
    vector4 current_point = evaluation_points[tid];

    current_point = project(&projection_matrix,
                            camera_plane_center,
                            current_point);

    evaluation_points[tid] = current_point;
  }
}
