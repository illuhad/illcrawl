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

#ifndef ENVIRONMENT
#define ENVIRONMENT

#include <string>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <map>
#include "qcl.hpp"

namespace illcrawl {

/// The illcrawl computing environment - initializes OpenCL, compiles kernels,
/// and assigns OpenCL devices to MPI processes.
class environment
{
public:
  /// Initialize the compute environment. Collective operation.
  /// \param argc Number of command line arguments
  /// \param argv command line arguments
  environment(int& argc, char**& argv)
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

    _global_ctx->global_register_source_file("reconstruction.cl",
                                            {"image_tile_based_reconstruction2D"});
    _global_ctx->global_register_source_file("volumetric_nn8_reconstruction.cl",
                                            {"volumetric_nn8_reconstruction",
                                             "finalize_volumetric_nn8_reconstruction"});
    _global_ctx->global_register_source_file("interpolation_tree.cl",
                                            {"tree_interpolation"});

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
                                              "unprocessed_quantity"
                                            });
    _global_ctx->global_register_source_file("integration.cl",
                                            {
                                               "runge_kutta_fehlberg",
                                               "construct_evaluation_points_over_camera_plane",
                                               "gather_integrand_evaluations"
                                            });

    determine_num_local_processes();

    _ctx = _global_ctx->device(static_cast<std::size_t>(_local_rank)
                               % _global_ctx->get_num_devices());

    boost::mpi::all_gather(get_communicator(), _ctx->get_device_name(), _device_names);

    std::string extensions;
    _ctx->get_supported_extensions(extensions);

    boost::mpi::all_gather(get_communicator(), extensions, _device_extensions);
  }

  ~environment()
  {
  }

  /// \return The OpenCL device context that has been assigned
  /// to this MPI process.
  qcl::device_context_ptr get_compute_device() const
  {
    return _ctx;
  }

  /// \return The global OpenCL context for this node.
  qcl::global_context_ptr get_compute_environment() const
  {
    return _global_ctx;
  }

  /// \return The MPI communicator
  boost::mpi::communicator get_communicator() const
  {
    return boost::mpi::communicator{};
  }

  /// \return A map that maps node names to a vector containing
  /// the MPI ranks of the processes on the specified node.
  const std::map<std::string, std::vector<int>>& get_node_rank_map() const
  {
    return _node_rank_map;
  }

  /// \return The names of the assigned OpenCL devices of all
  /// MPI processes.
  const std::vector<std::string>& get_device_names() const
  {
    return _device_names;
  }

  /// \return The supported extensions of the assigned OpenCL devices of all
  /// MPI processes.
  const std::vector<std::string>& get_device_extensions() const
  {
    return _device_extensions;
  }

  /// \return The MPI rank of the master process.
  static int get_master_rank() {return 0;}
private:

  /// Determines which MPI processes are on which node,
  /// and determines the number of local processes (i.e.
  /// the number of processes that run on the node
  /// of this process). Collective operation.
  void determine_num_local_processes()
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

  qcl::global_context_ptr _global_ctx;
  qcl::device_context_ptr _ctx;
  qcl::environment _env;

  int _num_global_processes;
  int _num_local_processes;
  int _local_rank;

  std::string _processor_name;

  boost::mpi::environment _mpi_environment;

  std::map<std::string, std::vector<int>> _node_rank_map;
  std::vector<std::string> _device_names;
  std::vector<std::string> _device_extensions;
};

}

#endif
