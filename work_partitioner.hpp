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

#ifndef WORK_PARTITIONER
#define WORK_PARTITIONER


#include <boost/mpi.hpp>
#include <memory>

namespace illcrawl {


class work_partitioner
{
public:
  virtual ~work_partitioner(){}

  /// create copy of object
  virtual std::unique_ptr<work_partitioner> clone() const = 0;

  /// Execute work partitioning algorithm
  /// \paran num_jobs The total number of parallel jobs
  virtual void run(std::size_t num_jobs) = 0;

  /// \return The index of the first job assigned to the calling process
  virtual std::size_t own_begin() const = 0;

  /// \return The first job index that is not assigned anymore to the calling
  /// process.
  virtual std::size_t own_end() const = 0;

  /// \return The number of jobs assigned to the calling process
  virtual std::size_t get_num_local_jobs() const = 0;

  /// \return The total number of jobs
  virtual std::size_t get_num_global_jobs() const = 0;

  /// \return The MPI communicator containing the processes among
  /// which the work has been distributed.
  virtual boost::mpi::communicator get_communicator() const = 0;

};

}

#endif
