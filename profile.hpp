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

template<class Partitioner,
         class Volumetric_reconstructor>
class distributed_mc_radial_profile
{
public:
  distributed_mc_radial_profile(const qcl::device_context_ptr& ctx,
                 const Partitioner& partitioner,
                 const math::vector3& profile_center,
                 math::scalar max_radius,
                 unsigned num_radii)
    : _ctx{ctx},
      _profile_center{profile_center},
      _max_radius{max_radius},
      _num_radii{num_radii},
      _profile(num_radii, 0),
      _partitioning{partitioner}
  {
    assert(ctx != nullptr);
    assert(num_radii > 0);
    // Calculate radii
    for(unsigned i = 0; i < num_radii; ++i)
      _radii.push_back(get_radius_of_shell(i));

  }

  void operator()(Volumetric_reconstructor& reconstructor,
                  const reconstruction_quantity::quantity& reconstructed_quantity,
                  math::scalar sample_density,
                  std::vector<math::scalar>& out_profile,
                  int root_process = 0)
  {
    std::fill(_profile.begin(), _profile.end(), 0.0);
    out_profile = std::vector<math::scalar>(_radii.size(), 0);

    reconstructor.purge_state();

    std::vector<std::size_t> overall_num_samples;

    run(std::vector<math::scalar>(_radii.size(), sample_density),
        reconstructor,
        reconstructed_quantity,
        out_profile,
        overall_num_samples,
        root_process);

    std::cout << "Redistributing samples..." << std::endl;
    // redo calculation and redistribute samples according to out_profile
    std::vector<math::scalar> sample_densities(_radii.size(), sample_density);

    // After the first run, overall_num_samples will contain the sample
    // number of exactly one run. The number of samples we want to generate
    // for each run is therefore given by the sum of all elements
    // in overall_num_samples.
    std::size_t samples_per_run = std::accumulate(overall_num_samples.begin(),
                                                  overall_num_samples.end(), 0);

    if(_partitioning.get_communicator().rank() == root_process)
    {
      // normalize out_profile
      math::scalar norm = std::accumulate(out_profile.begin(), out_profile.end(), 0.0);
      if(norm == 0)
        return;

      assert(out_profile.size() == _radii.size());
      for(std::size_t i = 0; i < _radii.size(); ++i)
      {
        //math::scalar weight = out_profile[i] * samples_per_run / norm;
        math::scalar shell_volume = get_shell_volume(i);
        std::size_t num_samples = out_profile[i] * samples_per_run / norm;
        sample_densities[i] = num_samples / shell_volume;
      }
    }

    // broadcast densities to all processes
    boost::mpi::broadcast(_partitioning.get_communicator(), sample_densities, root_process);

    run(sample_densities,
        reconstructor,
        reconstructed_quantity,
        out_profile,
        overall_num_samples,
        root_process);

  }

  const std::vector<math::scalar>& get_profile_radii() const
  {
    return _radii;
  }

private:
  void run(const std::vector<math::scalar>& sampling_densities,
           Volumetric_reconstructor& reconstructor,
           const reconstruction_quantity::quantity& reconstructed_quantity,
           std::vector<math::scalar>& result,
           std::vector<std::size_t>& overall_sample_counter,
           int root_process)
  {
    result.resize(_sample_points.size());
    std::fill(result.begin(), result.end(), 0.0);

    assert(sampling_densities.size() == _radii.size());

    // Make sure overall_num_samples has the right size if
    // this is the first call to run
    overall_sample_counter.resize(_radii.size(), 0.0);

    std::vector<std::size_t> total_samples_for_shell;
    std::size_t total_num_samples = estimate_num_samples(sampling_densities,
                                                         total_samples_for_shell);

    for(std::size_t i = 0; i < _radii.size(); ++i)
    {
      // Make sure each shell is sampled at least once
      // to prevent NaNs when dividing by the sample count
      // later on.
      if(total_samples_for_shell[i] == 0)
        total_samples_for_shell[i] = 1;
      // Add total_num_samples of this function call to the overall sample number counter
      overall_sample_counter[i] += total_samples_for_shell[i];
    }


    std::cout << "Total samples: " << total_num_samples << std::endl;

    // Determine local number of samples
    std::vector<std::size_t> local_number_of_samples;

    _partitioning.run(total_num_samples);

    std::size_t num_local_jobs = _partitioning.get_num_local_jobs();

    std::size_t global_sample_id = 0;
    for(std::size_t i = 0; i < _radii.size(); ++i)
    {
      // Calculate the intersection between the intervals
      // [partitioning.own_begin, partitioning.own_end]
      // and [global_sample_id, global_sample_id + total_samples_for_shell[i]]
      std::size_t interval_end = std::min(_partitioning.own_end(), global_sample_id + total_samples_for_shell[i]);
      std::size_t interval_begin = std::max(_partitioning.own_begin(), global_sample_id);
      std::size_t local_samples_for_shell = 0;
      if(interval_end > interval_begin)
        local_samples_for_shell = interval_end - interval_begin;

      local_number_of_samples.push_back(local_samples_for_shell);

      global_sample_id += total_samples_for_shell[i];
    }

    // Split the calculation into several batches
    // to make sure the number of samples is below the max batch size.
    // This is important to avoid running out of VRAM on the
    // device.
    std::size_t num_batches =
        std::ceil(static_cast<math::scalar>(num_local_jobs) / _max_batch_size);

    std::vector<device_scalar> evaluated_samples;
    for(std::size_t batch_id = 0; batch_id < num_batches; ++batch_id)
    {
      std::vector<std::size_t> num_samples_for_batch = local_number_of_samples;
      for(std::size_t i = 0; i < num_samples_for_batch.size(); ++i)
      {
        if(batch_id == num_batches - 1)
        {
          // Assign all remaining samples to the last batch (fixes truncation errors)
          assert(local_number_of_samples[i] >= (local_number_of_samples[i] / num_batches) * batch_id);
          num_samples_for_batch[i] = local_number_of_samples[i]
                                   - (local_number_of_samples[i] / num_batches) * batch_id;
        }
        else
          num_samples_for_batch[i] /= num_batches;
      }

      std::cout << "Number of samples in batch: "
                << std::accumulate(num_samples_for_batch.begin(), num_samples_for_batch.end(), 0)
                << std::endl;
      // Finally we can actually create the samples
      create_samples(num_samples_for_batch);
      // ...and evaluate them
      evaluate_samples(reconstructor, reconstructed_quantity, evaluated_samples);
      // Add samples to profile
      for(std::size_t shell = 0; shell < _radii.size(); ++shell)
      {
        // find out where the samples for this shell start and end
        std::size_t sample_points_begin = _first_sample_id_for_shell[shell];
        std::size_t sample_points_end = _sample_points.size();
        if(shell != _radii.size() - 1)
          sample_points_end = _first_sample_id_for_shell[shell + 1];

        assert(sample_points_end >= sample_points_begin);

        for(std::size_t sample = sample_points_begin; sample < sample_points_end; ++sample)
          _profile[shell] += static_cast<math::scalar>(evaluated_samples[sample]);
      }

      retrieve_global_profile(result, root_process);
      // Apply dV factor on root process
      if(_partitioning.get_communicator().rank() == root_process)
      {
        apply_volume_factor(overall_sample_counter, reconstructed_quantity, result);
      }
    }
  }

  void apply_volume_factor(const std::vector<std::size_t>& global_num_samples,
                           const reconstruction_quantity::quantity& reconstructed_quantity,
                           std::vector<math::scalar>& out_profile) const
  {
    for(std::size_t shell = 0; shell < _radii.size(); ++shell)
    {
      math::scalar shell_volume = get_shell_volume(shell);

      math::scalar dV = shell_volume / (global_num_samples[shell]);
      math::scalar unit_system_conversion =
          reconstructed_quantity.get_unit_converter().volume_conversion_factor();


      out_profile[shell] *= reconstructed_quantity.effective_volume_integration_dV(
            dV * unit_system_conversion, shell_volume * unit_system_conversion);
    }

  }

  std::size_t estimate_num_samples(const std::vector<math::scalar>& sampling_densities,
                                   std::vector<std::size_t>& samples_for_shell) const
  {
    samples_for_shell.clear();

    for(std::size_t i = 0; i < sampling_densities.size(); ++i)
      samples_for_shell.push_back(sampling_densities[i] * get_shell_volume(i));

    std::size_t total_num_samples = std::accumulate(samples_for_shell.begin(),
                                                    samples_for_shell.end(),
                                                    0);
    return total_num_samples;
  }

  void create_samples(const std::vector<std::size_t>& num_samples)
  {
    assert(num_samples.size() == _radii.size());

    std::size_t local_sample_id = 0;

    _first_sample_id_for_shell.clear();
    _sample_points.clear();
    for(std::size_t i = 0; i < _radii.size(); ++i)
    {
      math::scalar r_min = get_shell_start(i);
      math::scalar r_max = get_shell_end(i);
      assert(r_max > r_min);


      random::sampler::uniform_spherical_shell shell_sampler{r_min, r_max};

      _first_sample_id_for_shell.push_back(local_sample_id);
      for(std::size_t j = 0; j < num_samples[i]; ++j)
      {
        _sample_points.push_back(math::to_device_vector4(shell_sampler(_rng) + _profile_center));
        ++local_sample_id;
      }

    }

    // Create buffer on device and move sampling points to compute device
    _ctx->create_input_buffer<device_vector4>(_sampling_point_buffer, _sample_points.size());
    _ctx->memcpy_h2d(_sampling_point_buffer, _sample_points.data(), _sample_points.size());
  }

  void evaluate_samples(Volumetric_reconstructor& reconstructor,
                        const reconstruction_quantity::quantity& reconstructed_quantity,
                        std::vector<device_scalar>& reconstruction)
  {

    reconstructor.run(_sampling_point_buffer,
                      _sample_points.size(),
                      reconstructed_quantity);

    // Retrieve results
    reconstruction.resize(_sample_points.size());

    assert(_sample_points.size() == reconstructor.get_num_reconstructed_points());
    _ctx->memcpy_d2h(reconstruction.data(),
                     reconstructor.get_reconstruction(),
                     _sample_points.size());
  }

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

  random::random_number_generator _rng;

  const std::size_t _max_batch_size = 2000000;
};

}
}

#endif
