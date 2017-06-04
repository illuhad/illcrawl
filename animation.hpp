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


#ifndef ANIMATION
#define ANIMATION

#include <functional>
#include <cassert>
#include <boost/mpi.hpp>
#include <cmath>

#include "math.hpp"
#include "camera.hpp"
#include "volumetric_reconstruction.hpp"

namespace illcrawl {


class animation
{
public:
  using frame_renderer = std::function<void (const camera& cam, util::multi_array<device_scalar>& out)>;
  using camera_stepper = std::function<void (std::size_t frame_id, std::size_t num_frames, camera& cam)>;


  animation(const frame_renderer& renderer,
            const camera_stepper& stepper,
            const camera& initial_camera)
    : _renderer{renderer},
      _stepper{stepper},
      _cam{initial_camera}
  {}

  virtual ~animation(){}

  virtual void operator()(std::size_t num_frames, util::multi_array<device_scalar>& out)
  {
    (*this)(0, num_frames, num_frames, out);
  }

protected:
  /// \param first_frame The first frame of the frame range that should be
  /// renderered
  /// \param end_frame The frame after the last frame to be rendered
  /// \param total_frame_range The total number of frames that should be rendered
  /// \out The output, with each frame being a slice along the z-Axis of the
  /// multi array.
  void operator()(std::size_t first_frame,
                  std::size_t end_frame,
                  std::size_t total_frame_range,
                  util::multi_array<device_scalar>& out)
  {
    if(first_frame == end_frame)
      return;

    assert(end_frame > first_frame);

    out = util::multi_array<device_scalar>{
        _cam.get_num_pixels(0),
        _cam.get_num_pixels(1),
        end_frame - first_frame
    };

    util::multi_array<device_scalar> frame;
    for(std::size_t i = first_frame; i < end_frame; ++i)
    {
      _stepper(i, total_frame_range, _cam);
      _renderer(_cam, frame);

      assert(frame.get_extent_of_dimension(0) == _cam.get_num_pixels(0));
      assert(frame.get_extent_of_dimension(1) == _cam.get_num_pixels(1));

      // Copy rendered frame to result cube
      for(std::size_t y = 0; y < _cam.get_num_pixels(1); ++y)
        for(std::size_t x = 0; x < _cam.get_num_pixels(0); ++x)
        {
          std::size_t idx2 [] = {x,y};
          std::size_t idx3 [] = {x,y, i - first_frame};
          out[idx3] = frame[idx2];
        }
    }
  }

private:

  frame_renderer _renderer;
  camera_stepper _stepper;
  camera _cam;
};

template<class Partitioner>
class distributed_animation : public animation
{
public:
  distributed_animation(const Partitioner& partitioner,
                        const frame_renderer& renderer,
                        const camera_stepper& stepper,
                        const camera& initial_camera)
    : animation{renderer, stepper, initial_camera},
      _partitioner{partitioner}
  {}

  virtual void operator()(std::size_t num_frames, util::multi_array<device_scalar>& local_result) override
  {
    _partitioner.run(num_frames);

    animation::operator ()(_partitioner.own_begin(), _partitioner.own_end(), num_frames, local_result);
  }

  virtual ~distributed_animation(){}

  const Partitioner& get_partitioning() const
  {
    return _partitioner;
  }
private:
  Partitioner _partitioner;
};

namespace camera_movement {

class rotation_around_point
{
public:

  rotation_around_point(const math::vector3& center,
                        const math::vector3& axis,
                        const camera& cam,
                        math::scalar rotation_range_degree = 360.0)
    : _rotation_center(center),
      _rotation_axis(axis),
      _rotation_range{(2.0 * M_PI / 360.0) * rotation_range_degree},
      _initial_camera{cam}
  {
  }

  void operator()(std::size_t frame_id, std::size_t num_frames, camera& cam) const
  {
    math::scalar angle_per_frame = _rotation_range / static_cast<math::scalar>(num_frames);

    math::matrix3x3 rotation_matrix;
    math::matrix_create_rotation_matrix(&rotation_matrix,
                                        _rotation_axis,
                                        frame_id * angle_per_frame);

    // Calculate new camera position
    math::vector3 pos = _initial_camera.get_position();
    math::vector3 R = pos - _rotation_center;
    math::vector3 R_prime = math::matrix_vector_mult(rotation_matrix, R);
    cam.set_position(_rotation_center + R_prime);

    // Calculate new look_at vector
    math::vector3 look_at = _initial_camera.get_look_at();
    cam.set_look_at(math::matrix_vector_mult(rotation_matrix, look_at));
  }


  const math::vector3& get_rotation_axis() const
  {
    return _rotation_axis;
  }

  void set_rotation_axis(const math::vector3& axis)
  {
    _rotation_axis = axis;
  }

private:
  math::vector3 _rotation_center;
  math::vector3 _rotation_axis;
  const math::scalar _rotation_range;
  const camera _initial_camera;
};

class dual_axis_rotation_around_point
{
public:
  dual_axis_rotation_around_point(const math::vector3& center,
                                   const math::vector3& initial_axis,
                                   // The rotation axis will itself be rotated around the theta
                                   // axis
                                   const math::vector3& theta_axis,
                                   const camera& cam,
                                   math::scalar rotation_range_phi_degree = 360.0,
                                   math::scalar rotation_range_theta_degree = 360.0)
  : _rotation_around_point{center, initial_axis, cam, rotation_range_phi_degree},
    _initial_axis(initial_axis),
    _theta_axis(theta_axis),
    _theta_rotation_range{(2.0 * M_PI / 360.0) * rotation_range_theta_degree}
  {}

  void operator()(std::size_t frame_id, std::size_t num_frames, camera& cam)
  {
    math::scalar angle_per_frame = _theta_rotation_range / static_cast<math::scalar>(num_frames);

    math::matrix3x3 rotation_matrix;
    math::matrix_create_rotation_matrix(&rotation_matrix,
                                        _theta_axis,
                                        frame_id * angle_per_frame);

    _rotation_around_point.set_rotation_axis(math::matrix_vector_mult(rotation_matrix, _initial_axis));

    _rotation_around_point(frame_id, num_frames, cam);
  }

private:
  rotation_around_point _rotation_around_point;

  const math::vector3 _initial_axis;
  const math::vector3 _theta_axis;
  math::scalar _theta_rotation_range;
};

} // camera_movement

namespace animation_frame {

template<class Volumetric_reconstructor,
         class Integration_tolerance_type>
class integrated_projection
{
public:
  integrated_projection(const qcl::device_context_ptr& ctx,
              const reconstruction_quantity::quantity& reconstructed_quantity,
              math::scalar integration_depth,
              const Integration_tolerance_type& tol,
              Volumetric_reconstructor& reconstructor)
    : _ctx{ctx},
      _reconstructed_quantity{reconstructed_quantity},
      _integration_range{integration_depth},
      _tolerance{tol},
      _reconstructor{reconstructor}
  {
    assert(_ctx != nullptr);
  }

  void operator()(const camera& cam, util::multi_array<device_scalar>& out)
  {
    volumetric_integration<Volumetric_reconstructor> integrator{_ctx, cam};

    integrator.create_projection(_reconstructor,
                                 _reconstructed_quantity,
                                 _integration_range,
                                 _tolerance,
                                 out);
  }

private:
  qcl::device_context_ptr _ctx;
  const reconstruction_quantity::quantity& _reconstructed_quantity;

  math::scalar _integration_range;
  Integration_tolerance_type _tolerance;

  Volumetric_reconstructor _reconstructor;
};

}
}

#endif

