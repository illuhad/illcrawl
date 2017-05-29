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


#ifndef INTERPOLATION_CL
#define INTERPOLATION_CL


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

// f(r) = 1-r/R
scalar line_of_sight_integral(scalar smoothing_length,
                              scalar smoothing_length2,
                              scalar impact_parameter,
                              scalar impact_parameter2)
{
  scalar norm = 3.0f / (smoothing_length*smoothing_length2*M_PI_F);

  scalar s = sqrt(smoothing_length2 - impact_parameter2);
  return norm * 2.f * (s + impact_parameter2 / smoothing_length * log(impact_parameter / (smoothing_length + s)));
}

// f(r) = 1+(r/R)^4-2(r/R)^2
scalar line_of_sight_integral2(scalar R,
                              scalar R2,
                              scalar b,
                              scalar b2)
{

  scalar norm = 105.f / (32.f * R*R2*M_PI_F);

  scalar R4 = R2*R2;
  scalar s = sqrt(R2 - b2);
  scalar integral = 16.f/15.f * (R4-2.f*R2*b2+b2*b2) * s / R4;
  return 2.f * norm * integral;
}

scalar get_weight(vector2 coord,
                  scalar smoothing_length,
                  vector2 px_center,
                  scalar dx,
                  scalar dy,
                  vector2 pixel_min,
                  vector2 pixel_max)
{


  scalar impact_parameter2 = distance2(coord, px_center);
  scalar smoothing_length2 = smoothing_length * smoothing_length;

  if(impact_parameter2 < smoothing_length2)
  {

    vector2 smoothing_begin = coord;
    smoothing_begin -= smoothing_length;

    vector2 smoothing_end = coord;
    smoothing_end += smoothing_length;


    if(smoothing_begin.x >= pixel_min.x &&
       smoothing_end.x <= pixel_max.x &&
       smoothing_begin.y >= pixel_min.y &&
       smoothing_end.y <= pixel_max.y)
      return 1.f;
    else
    {
      return dx * dy * line_of_sight_integral2(smoothing_length,
                                              smoothing_length2,
                                              sqrt(impact_parameter2),
                                              impact_parameter2);
    }
  }

  return 0.0f;
}

#endif
