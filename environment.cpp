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

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include "environment.hpp"


namespace illcrawl {

environment::environment(int& argc, char**& argv)
  : _mpi_environment{argc, argv, boost::mpi::threading::funneled}
{

  boost::mpi::communicator comm = get_communicator();
  _num_global_processes = comm.size();

  this->_processor_name = boost::mpi::environment::processor_name();

  const cl::Platform& plat =
      _env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});

  _global_ctx =
      _env.create_global_context(plat, CL_DEVICE_TYPE_GPU);

  if (_global_ctx->get_num_devices() == 0)
  {
    throw std::runtime_error{"No OpenCL GPU devices found!"};
  }

  _global_ctx->global_register_source_file("projective_smoothing_reconstruction.cl",
                                           {"image_tile_based_reconstruction2D"});
  _global_ctx->global_register_source_file("volumetric_nn8_reconstruction.cl",
                                           {
                                             "volumetric_nn8_reconstruction",
                                             "finalize_volumetric_nn8_reconstruction"
                                           });
  _global_ctx->global_register_source_file("interpolation_tree.cl",
                                           {
                                             "tree_interpolation"
                                           });
  _global_ctx->global_register_source_file("dm_reconstruction_backend_brute_force.cl",
                                           {
                                             "dm_reconstruction_brute_force_smoothing"
                                           });
  _global_ctx->global_register_source_file("dm_reconstruction_backend_grid.cl",
                                           {
                                             "dm_reconstruction_grid_smoothing"
                                           });

  _global_ctx->global_register_source_file("quantities.cl",
                                           // Kernels inside quantities.cl
                                           {
                                             "luminosity_weighted_temperature",
                                             "xray_flux",
                                             "xray_photon_flux",
                                             "xray_spectral_flux",
                                             "identity",
                                             "mean_temperature",
                                             "chandra_xray_total_count_rate",
                                             "chandra_xray_spectral_count_rate",
                                             "unprocessed_quantity",
                                             "constant_quantity"
                                           });
  _global_ctx->global_register_source_file("integration.cl",
                                           {
                                             "runge_kutta_fehlberg",
                                             "construct_evaluation_points_over_camera_plane",
                                             "gather_integrand_evaluations"
                                           });
  _global_ctx->global_register_source_file("particle_grid.cl",
                                           {
                                             "grid3d_generate_sort_keys",
                                             "grid3d_determine_cells_begin",
                                             "grid3d_determine_cells_end"
                                           });
  _global_ctx->global_register_source_file("util.cl",
                                           {
                                             "util_create_sequence"
                                           });

  determine_num_local_processes();

  _ctx = _global_ctx->device(static_cast<std::size_t>(_local_rank)
                             % _global_ctx->get_num_devices());

  boost::mpi::all_gather(get_communicator(), _ctx->get_device_name(), _device_names);

  std::string extensions;
  _ctx->get_supported_extensions(extensions);

  boost::mpi::all_gather(get_communicator(), extensions, _device_extensions);
}

environment::~environment()
{
}

qcl::device_context_ptr environment::get_compute_device() const
{
  return _ctx;
}

qcl::global_context_ptr environment::get_compute_environment() const
{
  return _global_ctx;
}

/// \return The MPI communicator
boost::mpi::communicator environment::get_communicator() const
{
  return boost::mpi::communicator{};
}

const std::map<std::string, std::vector<int>>& environment::get_node_rank_map() const
{
  return _node_rank_map;
}

const std::vector<std::string>& environment::get_device_names() const
{
  return _device_names;
}

const std::vector<std::string>& environment::get_device_extensions() const
{
  return _device_extensions;
}


int environment::get_master_rank()
{
  return 0;
}

void environment::determine_num_local_processes()
{
  std::vector<std::string> names;
  boost::mpi::gather(get_communicator(), _processor_name, names, 0);

  _node_rank_map.clear();
  for(std::size_t i = 0; i < names.size(); ++i)
    _node_rank_map[names[i]].push_back(i);

  boost::mpi::broadcast(get_communicator(), _node_rank_map, 0);

  _num_local_processes = _node_rank_map[_processor_name].size();

  for(std::size_t i = 0; i < _node_rank_map[_processor_name].size(); ++i)
  {
    if(_node_rank_map[_processor_name][i] == get_communicator().rank())
    {
      _local_rank = i;
      return;
    }
  }
  throw std::runtime_error("Error in MPI environment: could not determine local node rank.");
}


}
