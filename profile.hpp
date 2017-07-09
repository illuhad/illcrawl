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

#ifndef PROFILE_HPP
#define PROFILE_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <boost/mpi.hpp>

#include "qcl.hpp"
#include "quantity.hpp"
#include "math.hpp"
#include "random.hpp"
#include "cl_types.hpp"

namespace illcrawl {
namespace analysis {

template<class Partitioner, class Volumetric_reconstructor>
class radial_profile
{
public:
  radial_profile(const qcl::device_context_ptr& ctx,
                 const Partitioner& partitioner,
                 const math::vector3& profile_center,
                 math::scalar max_radius,
                 unsigned num_radii,
                 math::scalar sample_density)
    : _ctx{ctx},
      _profile_center{profile_center},
      _max_radius{max_radius},
      _sample_density{sample_density},
      _num_radii{num_radii},
      _profile(num_radii, 0),
      _partitioning{partitioner}
  {
    assert(ctx != nullptr);
    assert(num_radii > 0);
    // Calculate radii
    for(unsigned i = 0; i < num_radii; ++i)
      _radii.push_back(get_radius_of_shell(i));

    // Calculate sampling points
    random::random_number_generator rng;

    math::scalar sphere_volume = 4./3. * M_PI * (max_radius * max_radius * max_radius);
    std::size_t total_num_samples = sample_density * sphere_volume;
    _partitioning.run(total_num_samples);

    math::scalar local_sample_density =
        static_cast<math::scalar>(_partitioning.get_num_local_jobs()) /
        static_cast<math::scalar>(total_num_samples) * sample_density;

    std::cout << "Total samples: " << total_num_samples << std::endl;
    std::size_t sample_id = 0;
    for(std::size_t i = 0; i < _radii.size(); ++i)
    {
      math::scalar r_min = get_shell_start(i);
      math::scalar r_max = get_shell_end(i);
      assert(r_max > r_min);

      random::sampler::uniform_spherical_shell shell_sampler{r_min, r_max};
      math::scalar shell_volume = get_shell_volume(r_min, r_max);

      // When calculating the local samples, we must take the maximum with 1
      // to exclude the sample number being truncated to 0 for all processes,
      // which can happen when there are a lot of parallel processes.
      // The minimum effective number of samples is therefore equal to the number
      // of parallel processes, regardless of specified sample density.
      std::size_t local_samples_for_shell =
          std::max(static_cast<std::size_t>(std::round(shell_volume * local_sample_density)),
                   std::size_t{1});

      _first_sample_id_for_shell.push_back(sample_id);
      for(std::size_t j = 0; j < local_samples_for_shell; ++j)
      {
        _sample_points.push_back(math::to_device_vector4(shell_sampler(rng) + _profile_center));
        ++sample_id;
      }
    }

    // Create buffer on device and move sampling points to compute device
    _ctx->create_input_buffer<device_vector4>(_sampling_point_buffer, _sample_points.size());
    _ctx->memcpy_h2d(_sampling_point_buffer, _sample_points.data(), _sample_points.size());
  }

  void operator()(Volumetric_reconstructor& reconstructor,
                  const reconstruction_quantity::quantity& reconstructed_quantity,
                  std::vector<math::scalar>& out_profile,
                  int root_process = 0)
  {
    reconstructor.purge_state();
    reconstructor.run(_sampling_point_buffer,
                      _sample_points.size(),
                      reconstructed_quantity);

    // Retrieve results
    std::vector<device_scalar> reconstruction(_sample_points.size());

    assert(_sample_points.size() == reconstructor.get_num_reconstructed_points());
    _ctx->memcpy_d2h(reconstruction.data(),
                     reconstructor.get_reconstruction(),
                     _sample_points.size());

    // Reset profile date
    std::fill(_profile.begin(), _profile.end(), 0.0);

    std::vector<std::size_t> local_num_samples;
    // Iterate over all shells (=radii) and calculate profile
    for(std::size_t shell = 0; shell < _radii.size(); ++shell)
    {
      // find out where the samples for this shell start and end
      std::size_t sample_points_begin = _first_sample_id_for_shell[shell];
      std::size_t sample_points_end = _sample_points.size();
      if(shell != _radii.size() - 1)
        sample_points_end = _first_sample_id_for_shell[shell + 1];

      // This must hold because we do not allow a process to have 0 samples
      // in a shell.
      assert(sample_points_end > sample_points_begin);

      for(std::size_t sample = sample_points_begin; sample < sample_points_end; ++sample)
        _profile[shell] += static_cast<math::scalar>(reconstruction[sample]);


      local_num_samples.push_back(sample_points_end - sample_points_begin);
    }
    // Find total number of samples
    std::vector<std::size_t> total_num_samples;
    boost::mpi::reduce(_partitioning.get_communicator(),
                       local_num_samples,
                       total_num_samples,
                       std::plus<std::size_t>(),
                       root_process);

    // Generate global profile by combining local profiles from
    // parallel processes
    retrieve_global_profile(out_profile, root_process);
    assert(out_profile.size() == _radii.size());

    if(_partitioning.get_communicator().rank() == root_process)
    {
      std::size_t num_processes = _partitioning.get_communicator().size();

      for(std::size_t shell = 0; shell < _radii.size(); ++shell)
      {
        math::scalar shell_volume = get_shell_volume(shell);

        math::scalar dV = shell_volume / (total_num_samples[shell] * num_processes);
        math::scalar unit_system_conversion =
            reconstructed_quantity.get_unit_converter().volume_conversion_factor();


        out_profile[shell] *= dV * unit_system_conversion;
        if(!reconstructed_quantity.is_integrated_quantity())
        {
          out_profile[shell] /= (shell_volume * unit_system_conversion);
        }
      }
    }

  }

  const std::vector<math::scalar>& get_profile_radii() const
  {
    return _radii;
  }

private:

  /// Collective operation
  void retrieve_global_profile(std::vector<math::scalar>& out, int root_process)
  {
    boost::mpi::reduce(_partitioning.get_communicator(),
                       _profile,
                       out,
                       std::plus<math::scalar>(),
                       root_process);
  }

  inline
  math::scalar get_shell_volume(unsigned shell) const
  {
    return get_shell_volume(get_shell_start(shell), get_shell_end(shell));
  }

  inline
  math::scalar get_shell_volume(math::scalar r_min, math::scalar r_max) const
  {
    assert(r_max >= r_min);
    return 4./3*M_PI*(r_max * r_max * r_max - r_min * r_min * r_min);
  }

  math::scalar get_radius_of_shell(unsigned bin_index) const
  {
    return bin_index * (_max_radius / _num_radii);
  }

  math::scalar get_shell_start(unsigned bin_index) const
  {
    if(bin_index == 0)
      return 0.0;

    return 0.5 * (get_radius_of_shell(bin_index - 1) + get_radius_of_shell(bin_index));
  }

  math::scalar get_shell_end(unsigned bin_index) const
  {
    return 0.5 * (get_radius_of_shell(bin_index + 1) + get_radius_of_shell(bin_index));
  }

  qcl::device_context_ptr _ctx;

  math::vector3 _profile_center;
  math::scalar _max_radius;

  math::scalar _sample_density;

  unsigned _num_radii;
  std::vector<math::scalar> _radii;
  std::vector<math::scalar> _profile;
  std::vector<device_vector4> _sample_points;


  cl::Buffer _sampling_point_buffer;

  Partitioner _partitioning;

  /// Holds the indices of the first sample belonging to a shell.
  /// I.e., the samples for shell i will start at at
  /// _first_sample_id_for_shell[i] in _sample_points.
  std::vector<std::size_t> _first_sample_id_for_shell;
};

}
}

#endif
