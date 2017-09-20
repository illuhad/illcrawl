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

#ifndef PROJECTIVE_SMOOTHING_BACKEND
#define PROJECTIVE_SMOOTHING_BACKEND

#include "reconstruction_backend.hpp"

namespace illcrawl {

/// Projective smoothing backends are reconstruction backends which
/// only take into account the x,y components of particles and evaluation
/// points (i.e., they work exclusively in the x,y plane) and
/// multiply a particle's weight (i.e. mass) not with the Smoothing kernel
/// W(x,y,z)  but with the (ideally analytically) projected smoothing
/// kernel W'(x,y) = Integral dz*W(x,y,z),
/// thereby creating a projected reconstruction very efficiently.
class projective_smoothing_backend : virtual public reconstruction_backend
{
public:
  virtual void setup_projected_particles(
                   const cl::Buffer& particles,
                   std::size_t num_particles,
                   const std::vector<cl::Buffer>& additional_datasets) = 0;
};

}

#endif
