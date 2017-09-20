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


#ifndef PROJECTIVE_SMOOTHING_GRID_RECONSTRUCTION_CL
#define PROJECTIVE_SMOOTHING_GRID_RECONSTRUCTION_CL


#define SPG_PROJECTION
#include "smoothing_particle_grid.cl"


__kernel void projective_smoothing_grid(
    __global int2* grid_cells,
    int3 num_grid_cells,
    vector3 grid_min_corner,
    vector3 grid_cell_sizes,

    __global vector4* particles,
    __global scalar* smoothing_lengths,
    scalar overall_max_smoothing_length,
    __global scalar* max_smoothing_lengths,

    int num_evaluation_points,
    __global vector4* evaluation_points_coordinates,
    __global scalar* results)
{
  spg_run_reconstruction(grid_cells,
                         num_grid_cells,
                         grid_min_corner,
                         grid_cell_sizes,
                         particles,
                         smoothing_lengths,
                         overall_max_smoothing_length,
                         max_smoothing_lengths,
                         num_evaluation_points,
                         evaluation_points_coordinates,
                         results);
}



#endif
