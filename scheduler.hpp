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

#ifndef SCHEDULER
#define SCHEDULER

#include <vector>
#include <boost/mpi.hpp>

namespace illcrawl {


class uniform_work_scheduler
{
public:
  uniform_work_scheduler(const boost::mpi::communicator& comm)
  : _comm(comm)
  {}

  void run(std::size_t num_jobs)
  {
    std::size_t nprocesses = static_cast<std::size_t>(_comm.size());

    this->_num_jobs = num_jobs;
    _domain_beginnings.clear();

    double jobs_per_proc = static_cast<double>(_num_jobs) / static_cast<double>(nprocesses);

    std::size_t current_beg = 0;
    for(std::size_t proc = 0; proc < nprocesses; ++proc)
    {
      _domain_beginnings.push_back(current_beg);
      current_beg = static_cast<std::size_t>(jobs_per_proc * static_cast<double>(proc + 1));
    }
  }

  std::size_t own_begin() const
  {
    return _domain_beginnings[_comm.rank()];
  }

  /// \return The id of the first job that
  /// does not belong to the local domain anymore.
  std::size_t own_end()  const
  {
    if(_comm.rank() == _comm.size() - 1)
      return _num_jobs;

    return _domain_beginnings[_comm.rank() + 1];
  }

  std::size_t get_num_local_jobs() const
  {
    return own_end() - own_begin();
  }
private:
  boost::mpi::communicator _comm;

  std::size_t _num_jobs;
  std::vector<std::size_t> _domain_beginnings;
};

}

#endif
