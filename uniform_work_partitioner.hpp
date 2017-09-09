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

#ifndef UNIFORM_WORK_PARTITIONER
#define UNIFORM_WORK_PARTITIONER

#include "work_partitioner.hpp"
#include <vector>

namespace illcrawl {


class uniform_work_partitioner : public work_partitioner
{
public:
  uniform_work_partitioner(const boost::mpi::communicator& comm);

  uniform_work_partitioner(const uniform_work_partitioner& other) = default;
  uniform_work_partitioner& operator=(const uniform_work_partitioner& other) = default;

  virtual std::unique_ptr<work_partitioner> clone() const override;

  virtual ~uniform_work_partitioner(){}

  virtual void run(std::size_t num_jobs) override;

  virtual std::size_t own_begin() const override;

  virtual std::size_t own_end()  const override;

  virtual std::size_t get_num_local_jobs() const override;

  virtual std::size_t get_num_global_jobs() const override;

  virtual boost::mpi::communicator get_communicator() const override;

private:
  boost::mpi::communicator _comm;

  std::size_t _num_jobs;
  std::vector<std::size_t> _domain_beginnings;
};


}

#endif
