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

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "math.hpp"

namespace illcrawl {


class camera
{
public:
  camera(const math::vector3& position,
         const math::vector3& look_at,
         math::scalar roll_angle,
         math::scalar screen_width,
         std::size_t num_pix_x,
         std::size_t num_pix_y)
    : _position(position),
      _look_at(look_at),
      _pixel_size{screen_width/static_cast<math::scalar>(num_pix_x)},
      _num_pixels{{num_pix_x, num_pix_y}},
      _roll_angle{roll_angle}
  {
    update_basis_vectors();
    update_min_position();
  }

  std::size_t get_num_pixels(std::size_t dim) const
  {
    return _num_pixels[dim];
  }

  math::vector3 get_pixel_coordinate(std::size_t x_index, std::size_t y_index) const
  {
    return _min_position
         + x_index * _pixel_size * _screen_basis_vector0
         + y_index * _pixel_size * _screen_basis_vector1;
  }

  math::vector3 get_pixel_coordinate(math::scalar x_index, math::scalar y_index) const
  {
    return _min_position
         + x_index * _pixel_size * _screen_basis_vector0
         + y_index * _pixel_size * _screen_basis_vector1;
  }

  math::scalar get_pixel_size() const
  {
    return _pixel_size;
  }

  const math::vector3& get_position() const
  {
    return _position;
  }

  const math::vector3& get_look_at() const
  {
    return _look_at;
  }

  const math::vector3& get_screen_basis_vector0() const
  {
    return _screen_basis_vector0;
  }

  const math::vector3& get_screen_basis_vector1() const
  {
    return _screen_basis_vector1;
  }

  void set_position(const math::vector3& pos)
  {
    this->_position = pos;
    update_min_position();
  }

  /// \param look_at A normalized vector indicating the
  /// direction in which the camera 'looks'
  void set_look_at(const math::vector3& look_at)
  {
    this->_look_at = look_at;
    update_basis_vectors();
    update_min_position();
  }

  const math::vector3& get_screen_min_position() const
  {
    return _min_position;
  }

  void rotate(const math::matrix3x3& rotation_matrix,
              const math::vector3& rotation_center)
  {
    math::vector3 R = _position - rotation_center;
    math::vector3 R_prime = math::matrix_vector_mult(rotation_matrix, R);
    this->_position = R_prime + rotation_center;

    this->_look_at = math::matrix_vector_mult(rotation_matrix, _look_at);

    this->_screen_basis_vector0 =
                     math::matrix_vector_mult(rotation_matrix, _screen_basis_vector0);
    this->_screen_basis_vector1 =
                     math::matrix_vector_mult(rotation_matrix, _screen_basis_vector1);

    update_min_position();
  }


private:
  void update_basis_vectors()
  {
    // Calculate screen basis vectors
    math::vector3 v1 {{0, 0, 1}};

    if (_look_at[0] == v1[0] &&
        _look_at[1] == v1[1] &&
        _look_at[2] == v1[2])
    {
      v1 = {{0, 1, 0}};
    }

    v1 = math::cross(v1, _look_at);
    math::vector3 v2 = math::cross(_look_at, v1);

    math::matrix3x3 roll_matrix;
    math::matrix_create_rotation_matrix(&roll_matrix, _look_at, _roll_angle);

    // Normalize vectors in case of rounding errors
    this->_screen_basis_vector0 =
        math::normalize(math::matrix_vector_mult(roll_matrix, v1));

    this->_screen_basis_vector1 =
        math::normalize(math::matrix_vector_mult(roll_matrix, v2));
  }

  void update_min_position()
  {
    _min_position = _position;
    _min_position -= (_num_pixels[0]/2.0) * _pixel_size * _screen_basis_vector0;
    _min_position -= (_num_pixels[1]/2.0) * _pixel_size * _screen_basis_vector1;
  }

  math::vector3 _position;
  math::vector3 _look_at;

  math::vector3 _screen_basis_vector0;
  math::vector3 _screen_basis_vector1;

  math::scalar _pixel_size;

  std::array<std::size_t, 2> _num_pixels;

  math::vector3 _min_position;

  math::scalar _roll_angle;
};

}

#endif
