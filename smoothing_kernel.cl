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


#ifndef SMOOTHING_KERNEL_CL
#define SMOOTHING_KERNEL_CL


#include "types.cl"


scalar quartic_polynomial3d(scalar r, scalar h)
{
  scalar h_inv = 1.f/h;
  // norm = 105/(32*h^3*pi)
  scalar norm = 1.04445f*h_inv*h_inv*h_inv;
  scalar q = fmin(r*h_inv, 1.f);
  scalar q2 = q*q;
  scalar q4 = q2*q2;

  return norm * (1.f + q4 - 2.f * q2);
}

scalar quartic_polynomial3d_cutoff_radius(scalar h)
{
  return h;
}

scalar quartic_polynomial3d_projection(scalar r,
                                       scalar h)
{
  scalar h_inv = 1.f/h;
  scalar h_inv3 = h_inv*h_inv*h_inv;
  // norm = 105/(32*h^3*pi)
  scalar norm = 1.04445f*h_inv3;

  scalar h4 = h*h*h*h;
  scalar s2 = fmax(h*h - r*r, 0.0f);

  // 16/15 ~ 1.06666666
  scalar integral = 1.066667f * s2*s2 * sqrt(s2) * h_inv * h_inv3;

  return 2.f * norm * integral;
}

/// See Monaghan (1992)
scalar cubic_spline3d(scalar r,
                      scalar h)
{
  scalar q = r/h;
  if(q > 2.f)
    return 0.0f;

  scalar norm = 1.f/(M_PI_F*h*h*h);

  if(q <= 1)
  {
    scalar q2 = q*q;
    scalar q3 = q*q2;
    return norm * (1.f - 1.5f*q2 + 0.75f*q3);
  }
  else
  {
    scalar two_minus_q = 2.f-q;
    return norm * 0.25f * two_minus_q*two_minus_q*two_minus_q;
  }
}

scalar cubic_spline3d_cutoff_radius(scalar h)
{
  return 2.f * h;
}


#ifdef USE_CUBIC_SPLINE
 #define CUTOFF_RADIUS(r) cubic_spline3d_cutoff_radius(r)
 #define SMOOTHING_KERNEL(r,h) cubic_spline3d(r,h)
 #define PROJECTED_SMOOTHING_KERNEL(b,h) quartic_polynomial3d_projection(b,h)
#else
 #define CUTOFF_RADIUS(r) quartic_polynomial3d_cutoff_radius(r)
 #define SMOOTHING_KERNEL(r,h) quartic_polynomial3d(r,h)
 #define PROJECTED_SMOOTHING_KERNEL(b,h) quartic_polynomial3d_projection(b,h)
#endif


#endif
