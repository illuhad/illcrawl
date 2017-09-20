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
#include <cassert>

#include "animation.hpp"
#include "projection.hpp"

namespace illcrawl {

/*********** Implementation of animation ***************/

animation::animation(const frame_renderer& renderer,
                     const camera_stepper& stepper,
                     const camera& initial_camera)
  : _renderer{renderer},
    _stepper{stepper},
    _cam{initial_camera}
{}

void
animation::operator()(std::size_t num_frames, util::multi_array<device_scalar>& out)
{
  (*this)(0, num_frames, num_frames, out);
}


void
animation::operator()(std::size_t first_frame,
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
    _renderer(_cam, i, total_frame_range, frame);

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

/******* Implementation of distributed_animation ***********/



distributed_animation::distributed_animation(const work_partitioner& partitioner,
                                             const frame_renderer& renderer,
                                             const camera_stepper& stepper,
                                             const camera& initial_camera)
  : animation{renderer, stepper, initial_camera},
    _partitioner{std::move(partitioner.clone())}
{}

void
distributed_animation::operator()(std::size_t num_frames,
                                  util::multi_array<device_scalar>& local_result)
{
  _partitioner->run(num_frames);

  animation::operator ()(_partitioner->own_begin(), _partitioner->own_end(), num_frames, local_result);
}

const work_partitioner&
distributed_animation::get_partitioning() const
{
  return *_partitioner;
}


namespace camera_movement {

/******** Implementation of rotation_around_point **********/

rotation_around_point::rotation_around_point(const math::vector3& center,
                                             const camera& cam,
                                             rotation_matrix_creator matrix_creator,
                                             math::scalar rotation_range_degree)
  : _rotation_center(center),
    _rotation_range{(2.0 * M_PI / 360.0) * rotation_range_degree},
    _initial_camera{cam},
    _rotation_creator{matrix_creator}
{
}

rotation_around_point::rotation_around_point(const math::vector3& center,
                                             const math::vector3& axis,
                                             const camera& cam,
                                             math::scalar rotation_range_degree)
  : _rotation_center(center),
    _rotation_range{(2.0 * M_PI / 360.0) * rotation_range_degree},
    _initial_camera{cam}
{
  _rotation_creator = [axis](math::scalar alpha) -> math::matrix3x3
  {
    math::matrix3x3 rotation;
    math::matrix_create_rotation_matrix(&rotation,
                                        axis,
                                        alpha);
    return rotation;
  };
}

void
rotation_around_point::operator()(std::size_t frame_id,
                                  std::size_t num_frames,
                                  camera& cam) const
{
  math::scalar angle_per_frame = _rotation_range / static_cast<math::scalar>(num_frames);

  math::matrix3x3 rotation_matrix = _rotation_creator(frame_id * angle_per_frame);

  // Reset camera to initial animation state
  cam = _initial_camera;
  // Now apply rotation to current state
  cam.rotate(rotation_matrix, _rotation_center);
}


void
rotation_around_point::set_rotation_matrix_creator(rotation_matrix_creator creator)
{
  _rotation_creator = creator;
}

/******** Implementation of dual_axis_rotation_around_point ***********/

dual_axis_rotation_around_point::dual_axis_rotation_around_point(const math::vector3& center,
                                                                 const math::vector3& phi_axis,
                                                                 const math::vector3& theta_axis,
                                                                 const camera& cam,
                                                                 math::scalar rotation_range_phi_degree,
                                                                 math::scalar rotation_range_theta_degree)
  : _rotation{center, phi_axis, cam, rotation_range_phi_degree}
{
  auto rotation_creator =
      [phi_axis, theta_axis, rotation_range_phi_degree, rotation_range_theta_degree](math::scalar alpha_phi)
  {
    math::scalar alpha_theta = alpha_phi * (rotation_range_theta_degree / rotation_range_phi_degree);
    return generate_rotation_matrix(alpha_phi, alpha_theta, phi_axis, theta_axis);
  };

  _rotation.set_rotation_matrix_creator(rotation_creator);
}

void
dual_axis_rotation_around_point::operator()(std::size_t frame_id,
                                            std::size_t num_frames,
                                            camera& cam)
{
  _rotation(frame_id, num_frames, cam);
}


math::matrix3x3
dual_axis_rotation_around_point::generate_rotation_matrix(math::scalar alpha_phi,
                                                          math::scalar alpha_theta,
                                                          const math::vector3& phi_axis,
                                                          const math::vector3& theta_axis)
{
  math::matrix3x3 M_phi, M_theta;
  math::matrix_create_rotation_matrix(&M_phi, phi_axis, alpha_phi);
  math::matrix_create_rotation_matrix(&M_theta, theta_axis, alpha_theta);

  return math::matrix_matrix_mult(M_phi, M_theta);
}


} // camera_movement

namespace frame_rendering {

/***************** Implementation of integrated_projection ***************/

integrated_projection::integrated_projection(const qcl::device_context_ptr& ctx,
                                             const reconstruction_quantity::quantity& reconstructed_quantity,
                                             math::scalar integration_depth,
                                             const integration::tolerance& tol,
                                             reconstructing_data_crawler* reconstructor)
  : _ctx{ctx},
    _reconstructed_quantity{reconstructed_quantity},
    _integration_range{integration_depth},
    _tolerance{tol},
    _reconstructor{reconstructor}
{
  assert(_reconstructor != nullptr);
  assert(_ctx != nullptr);
}

void
integrated_projection::operator()(const camera& cam,
                                  std::size_t frame_id,
                                  std::size_t num_frames,
                                  util::multi_array<device_scalar>& out)
{
  projection p{_ctx, cam};

  p.create_projection(*_reconstructor,
                      _reconstructed_quantity,
                      _integration_range,
                      _tolerance,
                      out);
}

/******** Implementation of multi_quantity_integrated_projection ********/


multi_quantity_integrated_projection::multi_quantity_integrated_projection(
    const qcl::device_context_ptr& ctx,
    math::scalar integration_depth,
    const integration::tolerance& tol,
    reconstructing_data_crawler* reconstructor,
    const quantity_generator& create_quantity)
  : _ctx{ctx},
    _integration_range{integration_depth},
    _tolerance{tol},
    _reconstructor{reconstructor},
    _create_quantity{create_quantity}
{
  assert(_ctx != nullptr);
  assert(_reconstructor != nullptr);
}



void
multi_quantity_integrated_projection::operator()(const camera& cam,
                                                 std::size_t frame_id,
                                                 std::size_t num_frames,
                                                 util::multi_array<device_scalar>& out)
{
  // This is a workaround to force the reconstructors to recalculate
  // the quantities - otherwise the reconstructor will assume that
  // nothing has changed and reuse the previous state.
  _reconstructor->purge_state();

  projection p{_ctx, cam};

  auto quantity = _create_quantity(cam, frame_id, num_frames);
  p.create_projection(*_reconstructor,
                      *quantity,
                      _integration_range,
                      _tolerance,
                      out);
}

/************* Implementation of single_quantity_generator **************/



single_quantity_generator::single_quantity_generator(
    const std::shared_ptr<reconstruction_quantity::quantity>& q)
  : _quantity{q}
{}

std::shared_ptr<reconstruction_quantity::quantity>
single_quantity_generator::operator()(const camera& cam,
                                      std::size_t frame_id,
                                      std::size_t num_frames) const
{
  return _quantity;
}


} // frame_rendering
} // illcrawl
