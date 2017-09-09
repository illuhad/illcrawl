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
  environment(int& argc, char**& argv);

  ~environment();

  /// \return The OpenCL device context that has been assigned
  /// to this MPI process.
  qcl::device_context_ptr get_compute_device() const;

  /// \return The global OpenCL context for this node.
  qcl::global_context_ptr get_compute_environment() const;

  /// \return The MPI communicator
  boost::mpi::communicator get_communicator() const;

  /// \return A map that maps node names to a vector containing
  /// the MPI ranks of the processes on the specified node.
  const std::map<std::string, std::vector<int>>& get_node_rank_map() const;

  /// \return The names of the assigned OpenCL devices of all
  /// MPI processes.
  const std::vector<std::string>& get_device_names() const;

  /// \return The supported extensions of the assigned OpenCL devices of all
  /// MPI processes.
  const std::vector<std::string>& get_device_extensions() const;

  /// \return The MPI rank of the master process.
  static int get_master_rank();
private:

  /// Determines which MPI processes are on which node,
  /// and determines the number of local processes (i.e.
  /// the number of processes that run on the node
  /// of this process). Collective operation.
  void determine_num_local_processes();

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
