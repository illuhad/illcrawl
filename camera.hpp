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

#include <cassert>

#include "math.hpp"

namespace illcrawl {

/// Represents a camera at a given position, look-at direction
/// and with a given resolution.
class camera
{
public:
  camera() = default;

  /// Construct object
  /// \param position The position of the center of the pixel
  /// screen of the camera
  /// \param look_at A normalized vector describing the direction
  /// the camera points to
  /// \param roll_angle The rotation state of the camera around
  /// the \c look_at axis in radians.
  /// \param screen_width The width of the screen in physical units
  /// \param num_pix_x The number of pixels in the x direction
  /// \param num_pix_y the number of pixels in the y direction
  camera(const math::vector3& position,
         const math::vector3& look_at,
         math::scalar roll_angle,
         math::scalar screen_width,
         std::size_t num_pix_x,
         std::size_t num_pix_y);

  /// \return The number of pixels either in x or in y direction
  /// \param dim Speicifies the axis (x or y). Must be 0 for the
  /// x axis, and 1 for the y axis. Other values are not allowed.
  std::size_t get_num_pixels(std::size_t dim) const;

  /// \return The coordinates of the specified pixel
  /// \param x_index The index in x direction of the pixel
  /// \param x_index The index in y direction of the pixel
  math::vector3 get_pixel_coordinate(std::size_t x_index, std::size_t y_index) const;

  /// \return The coordinates of the specified pixel
  /// \param x_index The index in x direction of the pixel
  /// \param x_index The index in y direction of the pixel
  math::vector3 get_pixel_coordinate(math::scalar x_index, math::scalar y_index) const;

  /// \return The size of a pixel (i.e. the side length of the pixel square)
  math::scalar get_pixel_size() const;

  /// \return The coordinates of the center of the pixel screen
  const math::vector3& get_position() const;

  /// \return A normalized vector describing the direction the camera
  /// points to
  const math::vector3& get_look_at() const;

  /// \return The basis vector for the x coordinate of the camera plane
  const math::vector3& get_screen_basis_vector0() const;

  /// \return The basis vector for the y coordinate of the camera plane
  const math::vector3& get_screen_basis_vector1() const;

  /// Set the position of the center of the pixel screen
  /// \param pos The new position
  void set_position(const math::vector3& pos);

  /// \param look_at A normalized vector indicating the
  /// direction in which the camera 'looks'
  void set_look_at(const math::vector3& look_at);

  /// \return The coordinates of the lower left corner of the pixel screen.
  const math::vector3& get_screen_min_position() const;

  /// Rotate the camera by applying a rotation matrix to the camera's
  /// position and look-at vector.
  /// \param rotation_matrix The rotation matrix
  /// \param rotation_center The point around which the camera shall be
  /// rotated
  void rotate(const math::matrix3x3& rotation_matrix,
              const math::vector3& rotation_center);

private:
  /// Calculates the basis vectors of the pixel screen.
  void update_basis_vectors();

  /// Calculates the lower left coordinate of the pixel screen.
  void update_min_position();

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
