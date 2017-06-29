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
#include "partitioner.hpp"

namespace illcrawl {
namespace spectrum {



class chandra_spectrum_quantity_generator
{
public:
  chandra_spectrum_quantity_generator(const qcl::device_context_ptr& ctx,
                   const unit_converter& converter,
                   const io::illustris_data_loader* data,
                   math::scalar redshift,
                   math::scalar luminosity_distance)
    : _ctx{ctx},
      _converter{converter},
      _data{data},
      _z{redshift},
      _luminosity_distance{luminosity_distance}
  {
    assert(ctx != nullptr);
  }

  std::shared_ptr<reconstruction_quantity::quantity> operator()(const camera& cam,
                                                                std::size_t frame_id,
                                                                std::size_t num_frames) const
  {
    math::scalar energy_bin_width = (chandra::arf::arf_max_energy() - chandra::arf::arf_min_energy())/
                                    static_cast<math::scalar>(num_frames);
    math::scalar current_energy = chandra::arf::arf_min_energy()
                                + frame_id * energy_bin_width;

    return std::make_shared<reconstruction_quantity::chandra_xray_spectral_count_rate>(
          _data,
          _converter,
          _ctx,
          _z,
          _luminosity_distance,
          current_energy,
          chandra::arf::arf_bin_width()
    );
  }

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
                math::scalar E_max)
    :_ctx{ctx},
     _converter{converter},
     _data{data},
     _z{redshift},
     _luminosity_distance{luminosity_distance},
     _min_energy{E_min},
     _max_energy{E_max}
  {
    assert(ctx != nullptr);
    assert(_max_energy >= _min_energy);
  }

  std::shared_ptr<reconstruction_quantity::quantity> operator()(const camera& cam,
                                                                std::size_t frame_id,
                                                                std::size_t num_frames) const
  {
    math::scalar energy_bin_width = (_max_energy - _min_energy)/
                                    static_cast<math::scalar>(num_frames);
    math::scalar current_energy = _min_energy
                                + frame_id * energy_bin_width;


    return std::make_shared<reconstruction_quantity::xray_spectral_flux>(
          _data,
          _converter,
          _ctx,
          _z,
          _luminosity_distance,
          current_energy,
          energy_bin_width
    );
  }

private:
  qcl::device_context_ptr _ctx;
  unit_converter _converter;
  const io::illustris_data_loader* _data;
  math::scalar _z;
  math::scalar _luminosity_distance;

  math::scalar _min_energy;
  math::scalar _max_energy;

};

template<class Volumetric_reconstructor,
         class Partitioner>
class spectrum_generator
{
public:
  spectrum_generator(const qcl::device_context_ptr& ctx,
                     const Partitioner& work_partitioner,
                     const Volumetric_reconstructor& reconstructor)
    :_ctx{ctx},
     _work_partitioner{work_partitioner},
     _reconstructor{reconstructor}
  {

  }

  ~spectrum_generator(){}

  template<class Integration_tolerance_type, class Quantity_creator>
  void operator()(const camera& cam,
                  const Integration_tolerance_type& tol,
                  math::scalar integration_depth,
                  const Quantity_creator& spectral_quantity_generator,
                  std::size_t num_energies,
                  util::multi_array<device_scalar>& local_result)
  {

    camera_movement::constant_state nonmoving_camera;

    frame_rendering::multi_quantity_integrated_projection
    <
        Volumetric_reconstructor,
        Integration_tolerance_type
    > renderer{_ctx, integration_depth, tol, _reconstructor, spectral_quantity_generator};

    distributed_animation<Partitioner> animation{
          _work_partitioner,
          renderer,
          nonmoving_camera,
          cam
    };

    animation(num_energies, local_result);
    _work_partitioner = animation.get_partitioning();
  }

  const Partitioner& get_partitioning() const
  {
    return _work_partitioner;
  }
private:
  qcl::device_context_ptr _ctx;
  Partitioner _work_partitioner;

  Volumetric_reconstructor _reconstructor;
};



}
}

#endif
