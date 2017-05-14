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


#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <array>
#include <cmath>

namespace illcrawl {
namespace integration {

template<class T>
class absolute_tolerance
{
public:
  absolute_tolerance(const T& tol)
    : _tol{tol}
  {}

  inline
  T get_absolute_tolerance(const T& integration_state) const
  {
    return _tol;
  }

private:
  T _tol;
};

template<class T>
class relative_tolerance
{
public:
  relative_tolerance(const T& tol)
    : _tol{tol}
  {}

  inline
  T get_absolute_tolerance(const T& integration_state) const
  {
    return _tol * integration_state;
  }

private:
  T _tol;
};

/// Solves dy/dz = f(z)
template<class T,
         class Coordinate_type>
class runge_kutta_fehlberg
{
public:
  runge_kutta_fehlberg(Coordinate_type initial_position = Coordinate_type{},
                       T initial_state = T{},
                       T first_evaluation = T{},
                       Coordinate_type initial_step_size = 1.0)
    : _state{initial_state},
      _interval_start_evaluation{first_evaluation},
      _step_size{initial_step_size},
      _current_position{initial_position}
  {}

  using evaluation_coordinates = std::array<Coordinate_type, 4>;
  using integrand_values = std::array<T, 4>;
  static constexpr std::size_t required_num_evaluations = 4;

  Coordinate_type get_position() const
  {
    return _current_position;
  }

  T get_state() const
  {
    return _state;
  }

  Coordinate_type get_step_size() const
  {
    return _step_size;
  }

  void obtain_next_step_coordinates(evaluation_coordinates& next_step_coordinates) const
  {
    // k1 will be provided by the last evaluation of the previous interval
    // next_step_coordinates[0] = _current_position;

    // We do not need k2 since we are only interested in the equation
    // dy/dz =f(z) and not dy/dz=f(y,z)
    //next_step_coordinates[1] = _current_position + 1./4. * _step_size;
    next_step_coordinates[0] = _current_position + 3./8. * _step_size;
    next_step_coordinates[1] = _current_position + 12./13. * _step_size;
    next_step_coordinates[2] = _current_position + _step_size;
    next_step_coordinates[3] = _current_position + 1./2. * _step_size;
  }

  template<class Tolerance_type>
  void advance(const integrand_values& values,
               const Tolerance_type& tolerance,
               Coordinate_type integration_end)
  {
    T delta4 =
           + 25./216.    * _interval_start_evaluation
           + 1408./2565. * values[0]
           + 2197./4101. * values[1]
           - 1./5.       * values[2];

    T delta5 =
           + 16./135.      * _interval_start_evaluation
           + 6656./12825.  * values[0]
           + 28561./56430. * values[1]
           - 9./50.        * values[2]
           + 2./55.        * values[3];

    delta4 *= _step_size;
    delta5 *= _step_size;

    T estimate4 = _state + delta4;
    T estimate5 = _state + delta5;

    _current_position += _step_size;


    T s = 2.0;
    if(estimate4 != estimate5)
    {
      T error = std::abs(estimate5 - estimate4);
      T absolute_tolerance = tolerance.get_absolute_tolerance(_state / _current_position);
      s = std::pow(absolute_tolerance * _step_size / (2 * error), 1./4.);
    }

    Coordinate_type new_step_size = s * _step_size;

    if(new_step_size < minimum_stepsize)
    {
      new_step_size = minimum_stepsize;
      s = new_step_size / _step_size;
    }

    if(s < 0.95)
    {
    // Reject approximation, go back to old position
      _current_position -= _step_size;
    }
    else
    {
      // Accept approximation
      _state = estimate4;
      _interval_start_evaluation = values[3];
    }

    _step_size = new_step_size;

    if(_current_position + _step_size > integration_end)
      // The epsilon's job is to make sure that the condition
      // get_position() < integration range turns false and a integration loop
      // does not turn into an infinite loop.
      _step_size = integration_end - _current_position + epsilon;
  }

private:
  static constexpr Coordinate_type minimum_stepsize = 0.2;
  static constexpr Coordinate_type epsilon = 0.01;

  T _state;
  T _interval_start_evaluation;

  Coordinate_type _step_size;
  Coordinate_type _current_position;
};

}
}

#endif
