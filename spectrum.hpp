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


#ifndef SPECTRUM_HPP
#define SPECTRUM_HPP

#include "animation.hpp"
#include "work_partitioner.hpp"
#include "integration.hpp"

namespace illcrawl {
namespace spectrum {

class chandra_spectrum_quantity_generator
{
public:
  chandra_spectrum_quantity_generator(const qcl::device_context_ptr& ctx,
                   const unit_converter& converter,
                   const io::illustris_data_loader* data,
                   math::scalar redshift,
                   math::scalar luminosity_distance);

  std::shared_ptr<reconstruction_quantity::quantity> operator()(const camera& cam,
                                                                std::size_t frame_id,
                                                                std::size_t num_frames) const;
private:
  qcl::device_context_ptr _ctx;
  unit_converter _converter;
  const io::illustris_data_loader* _data;
  math::scalar _z;
  math::scalar _luminosity_distance;

};


class xray_spectrum_quantity_generator
{
public:
  xray_spectrum_quantity_generator(const qcl::device_context_ptr& ctx,
                const unit_converter& converter,
                const io::illustris_data_loader* data,
                math::scalar redshift,
                math::scalar luminosity_distance,
                math::scalar E_min,
                math::scalar E_max);

  std::shared_ptr<reconstruction_quantity::quantity> operator()(const camera& cam,
                                                                std::size_t frame_id,
                                                                std::size_t num_frames) const;
private:
  qcl::device_context_ptr _ctx;
  unit_converter _converter;
  const io::illustris_data_loader* _data;
  math::scalar _z;
  math::scalar _luminosity_distance;

  math::scalar _min_energy;
  math::scalar _max_energy;

};


class spectrum_generator
{
public:
  spectrum_generator(const qcl::device_context_ptr& ctx,
                     reconstructing_data_crawler* reconstructor,
                     const work_partitioner& partitioner);

  ~spectrum_generator(){}

  template<class Quantity_creator>
  void operator()(const camera& cam,
                  const integration::tolerance& tol,
                  math::scalar integration_depth,
                  const Quantity_creator& spectral_quantity_generator,
                  std::size_t num_energies,
                  util::multi_array<device_scalar>& local_result)
  {
    _reconstructor->purge_state();

    camera_movement::constant_state nonmoving_camera;

    frame_rendering::multi_quantity_integrated_projection renderer{
      _ctx,
      integration_depth,
      tol,
      _reconstructor,
      spectral_quantity_generator
    };

    distributed_animation animation{
          *_partitioner,
          renderer,
          nonmoving_camera,
          cam
    };

    animation(num_energies, local_result);
    _partitioner = std::move(animation.get_partitioning().clone());
  }

  const work_partitioner& get_partitioning() const;
private:
  qcl::device_context_ptr _ctx;
  std::unique_ptr<work_partitioner> _partitioner;
  reconstructing_data_crawler* _reconstructor;
};



}
}

#endif
