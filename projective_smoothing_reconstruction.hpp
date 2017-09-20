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


#ifndef PROJECTIVE_SMOOTHING_RECONSTRUCTION_HPP
#define PROJECTIVE_SMOOTHING_RECONSTRUCTION_HPP

#include "projective_smoothing_backend.hpp"
#include "camera.hpp"

namespace illcrawl {
namespace reconstruction_backends {
namespace dm {

class projective_smoothing : public reconstruction_backend
{
public:

  projective_smoothing(const qcl::device_context_ptr& ctx,
                       const camera& cam,
                       math::scalar max_integration_depth,
                       std::unique_ptr<projective_smoothing_backend> smoothing_backend);

  virtual std::vector<H5::DataSet> get_required_additional_datasets() const final override;

  virtual const cl::Buffer& retrieve_results() final override;

  virtual std::string get_backend_name() const final override;

  virtual void init_backend(std::size_t blocksize) final override;

  virtual void setup_particles(const std::vector<particle>& particles,
                               const std::vector<cl::Buffer>& additional_dataset) final override;

  virtual void setup_evaluation_points(const cl::Buffer& evaluation_points,
                                       std::size_t num_points) final override;

  virtual void run() final override;

  virtual ~projective_smoothing(){}

  void set_camera(const camera& cam);

  void set_integration_depth(math::scalar depth);

private:
  static constexpr std::size_t local_size = 512;

  qcl::device_context_ptr _ctx;
  camera _cam;

  std::unique_ptr<projective_smoothing_backend> _backend;

  math::scalar _max_integration_depth;

  void project_evaluation_points(const camera& cam,
                                 const cl::Buffer& evaluation_points,
                                 std::size_t num_evaluation_points) const;

  void project_particles(const camera& cam,
                         const cl::Buffer& particles,
                         std::size_t num_particles,
                         math::scalar max_projection_distance) const;

};

}
}
}

#endif
