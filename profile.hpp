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
#include <memory>
#include <boost/mpi.hpp>

#include "qcl.hpp"
#include "quantity.hpp"
#include "math.hpp"
#include "random.hpp"
#include "cl_types.hpp"
#include "reconstructing_data_crawler.hpp"
#include "work_partitioner.hpp"

namespace illcrawl {
namespace analysis {


class distributed_mc_radial_profile
{
public:
  distributed_mc_radial_profile(const qcl::device_context_ptr& ctx,
                 const work_partitioner& partitioner,
                 const math::vector3& profile_center,
                 math::scalar max_radius,
                 unsigned num_radii);

  void operator()(reconstructing_data_crawler& reconstructor,
                  const reconstruction_quantity::quantity& reconstructed_quantity,
                  math::scalar sample_density,
                  std::vector<math::scalar>& out_profile,
                  int root_process = 0);

  /// \return The evaluation radii of the profile in proper units
  std::vector<math::scalar> get_proper_profile_radii() const;


  /// \return The evaluation radii of the profile in comoving units
  const std::vector<math::scalar>& get_comoving_profile_radii() const;
private:
  /// Generates sample numbers for each radius such
  /// that the error remains roughly constant for each radius.
  /// Since the error E scales with V/sqrt(N), it follows that the sample
  /// number N ~ V^2
  void generate_num_samples(std::vector<std::size_t>& num_samples,
                            math::scalar mean_density) const;

  void run(const std::vector<std::size_t>& num_samples,
           reconstructing_data_crawler& reconstructor,
           const reconstruction_quantity::quantity& reconstructed_quantity,
           std::vector<math::scalar>& result,
           int root_process);

  void apply_volume_factor(const std::vector<std::size_t>& global_num_samples,
                           const reconstruction_quantity::quantity& reconstructed_quantity,
                           std::vector<math::scalar>& out_profile) const;


  void create_samples(const std::vector<std::size_t>& num_samples);

  void evaluate_samples(reconstructing_data_crawler& reconstructor,
                        const reconstruction_quantity::quantity& reconstructed_quantity,
                        std::vector<device_scalar>& reconstruction);
  /// Collective operation
  void retrieve_global_profile(std::vector<math::scalar>& out, int root_process);

  inline
  math::scalar get_shell_volume(unsigned shell) const;

  inline
  math::scalar get_shell_volume(math::scalar r_min, math::scalar r_max) const;

  math::scalar get_radius_of_shell(unsigned bin_index) const;
  math::scalar get_shell_start(unsigned bin_index) const;
  math::scalar get_shell_end(unsigned bin_index) const;

  qcl::device_context_ptr _ctx;

  math::vector3 _profile_center;
  math::scalar _max_radius;


  unsigned _num_radii;
  std::vector<math::scalar> _radii;
  std::vector<math::scalar> _profile_sum_state;
  std::vector<device_vector4> _sample_points;


  cl::Buffer _sampling_point_buffer;

  std::unique_ptr<work_partitioner> _partitioning;

  /// Holds the indices of the first sample belonging to a shell.
  /// I.e., the samples for shell i will start at at
  /// _first_sample_id_for_shell[i] in _sample_points.
  std::vector<std::size_t> _first_sample_id_for_shell;

  random::random_number_generator _rng;

  const std::size_t _max_batch_size = 2000000;

  math::scalar _proper_units_conversion_factor = 1.0;
};

}
}

#endif
