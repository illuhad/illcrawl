#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <array>
#include <cmath>

namespace illcrawl {
namespace integration {

/// Solves dy/dz = f(z)
template<class T, class Coordinate_type>
class runge_kutta_fehlberg
{
public:


  runge_kutta_fehlberg(Coordinate_type initial_position = Coordinate_type{},
                       T initial_state = T{},
                       Coordinate_type initial_step_size = 1.0)
    : _state{initial_state},
      _current_position{initial_position},
      _step_size{initial_step_size}
  {}

  using evaluation_coordinates = std::array<Coordinate_type, 6>;
  using integrand_values = std::array<T, 6>;
  static constexpr std::size_t required_num_evaluations = 6;

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
    next_step_coordinates[0] = _current_position;
    next_step_coordinates[1] = _current_position + 1./4. * _step_size;
    next_step_coordinates[2] = _current_position + 3./8. * _step_size;
    next_step_coordinates[3] = _current_position + 12./13. * _step_size;
    next_step_coordinates[4] = _current_position + _step_size;
    next_step_coordinates[5] = _current_position + 1./2. * _step_size;
  }

  void advance(const integrand_values& values,
               T tolerance,
               Coordinate_type integration_end)
  {
    T delta4 =
           + 25./216.    * values[0]
           + 1408./2565. * values[2]
           + 2197./4101. * values[3]
           - 1./5.       * values[4];

    T delta5 =
           + 16./135.      * values[0]
           + 6656./12825.  * values[2]
           + 28561./56430. * values[3]
           - 9./50.        * values[4]
           + 2./55.        * values[5];

    delta4 *= _step_size;
    delta5 *= _step_size;

    T estimate4 = _state + delta4;
    T estimate5 = _state + delta5;

    _current_position += _step_size;


    T s = 2.0;
    if(estimate4 != estimate5)
    {
      T error = std::abs(estimate5 - estimate4);
      s = std::pow(tolerance * _step_size / (2 * error), 1./4.);
    }

    if(s < 0.95)
    // Reject approximation, go back to old position
      _current_position -= _step_size;
    else
      _state = estimate4;

    _step_size *= s;

    if(_step_size < 0.1)
      _step_size = 0.1;
    /*
    if(_current_position + _step_size > integration_end)
      _step_size = integration_end - _current_position;*/
  }

private:

  T _state;
  Coordinate_type _step_size;
  Coordinate_type _current_position;
};

}
}

#endif
