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

#include "profile.hpp"

namespace illcrawl {
namespace analysis {

distributed_mc_radial_profile::distributed_mc_radial_profile(
                              const qcl::device_context_ptr& ctx,
                              const work_partitioner& partitioner,
                              const math::vector3& profile_center,
                              math::scalar max_radius,
                              unsigned num_radii)
  : _ctx{ctx},
    _profile_center{profile_center},
    _max_radius{max_radius},
    _num_radii{num_radii},
    _profile_sum_state(num_radii, 0),
    _partitioning{std::move(partitioner.clone())}
{
  assert(ctx != nullptr);
  assert(num_radii > 0);
  // Calculate radii
  for(unsigned i = 0; i < num_radii; ++i)
    _radii.push_back(get_radius_of_shell(i));

}

void
distributed_mc_radial_profile::operator()(reconstructing_data_crawler& reconstructor,
                                          const reconstruction_quantity::quantity& reconstructed_quantity,
                                          math::scalar sample_density,
                                          std::vector<math::scalar>& out_profile,
                                          int root_process)
{
  std::fill(_profile_sum_state.begin(), _profile_sum_state.end(), 0.0);
  out_profile = std::vector<math::scalar>(_radii.size(), 0);

  reconstructor.purge_state();

  std::vector<std::size_t> num_samples;
  // Calculate the number of required samples
  // for each shell to achieve the given mean density
  this->generate_num_samples(num_samples, sample_density);

  run(num_samples,
      reconstructor,
      reconstructed_quantity,
      out_profile,
      root_process);


}

const std::vector<math::scalar>&
distributed_mc_radial_profile::get_profile_radii() const
{
  return _radii;
}

void
distributed_mc_radial_profile::generate_num_samples(std::vector<std::size_t>& num_samples,
                                                    math::scalar mean_density) const
{
  assert(_radii.size() > 0);

  num_samples = std::vector<std::size_t>(_radii.size());

  std::vector<math::scalar> sampling_scaling(_radii.size());

  for(std::size_t i = 0; i < _radii.size(); ++i)
  {
    math::scalar V = this->get_shell_volume(i);
    sampling_scaling[i] = V*V;
  }

  // Calculate normalization
  math::scalar expected_num_samples = 0;
  for(std::size_t i = 0; i < _radii.size(); ++i)
    expected_num_samples += this->get_shell_volume(i) * mean_density;

  math::scalar norm = static_cast<math::scalar>(expected_num_samples) /
                      std::accumulate(sampling_scaling.begin(),
                                      sampling_scaling.end(),
                                      0.0);


  // Normalize and set result
  for(std::size_t i = 0; i < _radii.size(); ++i)
  {
    num_samples[i] = static_cast<std::size_t>(norm * sampling_scaling[i]);
    if(num_samples[i] == 0)
      num_samples[i] = 1;
  }
}


void
distributed_mc_radial_profile::run(const std::vector<std::size_t>& num_samples,
                                   reconstructing_data_crawler& reconstructor,
                                   const reconstruction_quantity::quantity& reconstructed_quantity,
                                   std::vector<math::scalar>& result,
                                   int root_process)
{
  result.resize(_radii.size());
  std::fill(result.begin(), result.end(), 0.0);

  assert(num_samples.size() == _radii.size());

  for(std::size_t i = 0; i < _radii.size(); ++i)
  {
    // Make sure each shell is sampled at least once
    // to prevent NaNs when dividing by the sample count
    // later on.
    assert(num_samples[i] > 0);
  }
  std::size_t total_num_samples = std::accumulate(num_samples.begin(),
                                                  num_samples.end(),
                                                  0);

  std::cout << "Total samples: " << total_num_samples << std::endl;


  _partitioning->run(total_num_samples);

  // Determine local number of samples
  std::vector<std::size_t> local_num_samples;
  std::size_t num_local_jobs = _partitioning->get_num_local_jobs();

  std::size_t global_sample_id = 0;
  for(std::size_t i = 0; i < _radii.size(); ++i)
  {
    // Calculate the intersection between the intervals
    // [partitioning.own_begin, partitioning.own_end]
    // and [global_sample_id, global_sample_id + num_samples[i]]
    std::size_t interval_end = std::min(_partitioning->own_end(), global_sample_id + num_samples[i]);
    std::size_t interval_begin = std::max(_partitioning->own_begin(), global_sample_id);
    std::size_t local_samples_for_shell = 0;
    if(interval_end > interval_begin)
      local_samples_for_shell = interval_end - interval_begin;

    local_num_samples.push_back(local_samples_for_shell);

    global_sample_id += num_samples[i];
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
    std::vector<std::size_t> num_samples_for_batch = local_num_samples;
    for(std::size_t i = 0; i < num_samples_for_batch.size(); ++i)
    {
      if(batch_id == num_batches - 1)
      {
        // Assign all remaining samples to the last batch (fixes truncation errors)
        assert(local_num_samples[i] >= (local_num_samples[i] / num_batches) * batch_id);
        num_samples_for_batch[i] = local_num_samples[i]
            - (local_num_samples[i] / num_batches) * batch_id;
      }
      else
        num_samples_for_batch[i] /= num_batches;
    }

    std::cout << "Number of samples in batch: "
              << std::accumulate(num_samples_for_batch.begin(), num_samples_for_batch.end(), 0)
              << std::endl;
    // Finally we can actually create the samples. This creates
    // the samples and fills \c _first_sample_id_for_shell with
    // the offsets where samples for the shells begin
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
        _profile_sum_state[shell] += static_cast<math::scalar>(evaluated_samples[sample]);
    }

    retrieve_global_profile(result, root_process);
    // Apply dV factor on root process
    if(_partitioning->get_communicator().rank() == root_process)
    {
      apply_volume_factor(num_samples, reconstructed_quantity, result);
    }
  }
}

void
distributed_mc_radial_profile::apply_volume_factor(const std::vector<std::size_t>& global_num_samples,
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



void
distributed_mc_radial_profile::create_samples(const std::vector<std::size_t>& num_samples)
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

void
distributed_mc_radial_profile::evaluate_samples(reconstructing_data_crawler& reconstructor,
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
void
distributed_mc_radial_profile::retrieve_global_profile(std::vector<math::scalar>& out, int root_process)
{
  boost::mpi::reduce(_partitioning->get_communicator(),
                     _profile_sum_state,
                     out,
                     std::plus<math::scalar>(),
                     root_process);
}

inline math::scalar
distributed_mc_radial_profile::get_shell_volume(unsigned shell) const
{
  return get_shell_volume(get_shell_start(shell), get_shell_end(shell));
}

inline math::scalar
distributed_mc_radial_profile::get_shell_volume(math::scalar r_min, math::scalar r_max) const
{
  assert(r_max >= r_min);
  return 4./3*M_PI*(r_max * r_max * r_max - r_min * r_min * r_min);
}

math::scalar
distributed_mc_radial_profile::get_radius_of_shell(unsigned bin_index) const
{
  return bin_index * (_max_radius / _num_radii);
}

math::scalar
distributed_mc_radial_profile::get_shell_start(unsigned bin_index) const
{
  if(bin_index == 0)
    return 0.0;

  return 0.5 * (get_radius_of_shell(bin_index - 1) + get_radius_of_shell(bin_index));
}

math::scalar
distributed_mc_radial_profile::get_shell_end(unsigned bin_index) const
{
  return 0.5 * (get_radius_of_shell(bin_index + 1) + get_radius_of_shell(bin_index));
}



}
}
