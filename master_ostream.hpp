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

#include <ostream>
#include <mpi.h>

namespace illcrawl {
namespace util {

/// This class implements an ostream that will only write to the stream
/// on the master level. Calls from other processes will be ignored.
/// E.g. the following code will print "Hello world" only the process
/// of rank 0 in MPI_COMM_WORLD:
///
/// \code{.cpp}
/// master_ostream stream(std::cout, 0);
/// stream << "Hello World!\n";
/// \endcode
class master_ostream
{
public:
  /// Initializes the object. Collective on MPI_COMM_WORLD.
  /// \param ostr The output stream that shall be used. The reference
  /// must remain valid throughout the existence of the master_ostream
  /// object.
  /// \param master_rank The rank of the process (in MPI_COMM_WORLD)
  /// on which data shall be written.

  master_ostream(std::ostream& ostr,
                 const MPI_Comm& comm = MPI_COMM_WORLD,
                 int master_rank = 0)
    : _ostr{ostr}, _master_rank{master_rank}
  {
    MPI_Comm_rank(comm, &_rank);
  }

  /// Conversion to std::ostream&
  operator std::ostream& ()
  {
    return _ostr;
  }

  operator const std::ostream& () const
  {
    return _ostr;
  }

  /// Return the master rank.
  int get_master_rank() const
  {
    return _master_rank;
  }

  /// Get the process rank
  int get_rank() const
  {
    return _rank;
  }

  /// \return whether the calling process is the master process
  inline bool is_master_process() const
  {
    return _rank == _master_rank;
  }

  typedef std::ostream& (*io_manip_type)(std::ostream&);
private:
  std::ostream& _ostr;
  int _master_rank;
  int _rank;
};

/// Implements \c operator<<.
/// Only calls from the process specified as \c master_process during
/// the construction of the \c master_ostream object will lead to output
/// being written to the output stream. Calls from other processes will
/// be ignored. This call is not collective, but obviously at least
/// the specified master process has to call the function if any
/// output is to be written at all.

template<class T>
master_ostream& operator<<(master_ostream& ostr, const T& msg)
{
  if (ostr.get_rank() == ostr.get_master_rank())
    (std::ostream&)ostr << msg;

  return ostr;
}

/// This version enables the use of io manips
master_ostream& operator<<(master_ostream& ostr,
        master_ostream::io_manip_type io_manip)
{
  if (ostr.get_rank() == ostr.get_master_rank())
    io_manip((std::ostream&)ostr);

  return ostr;
}

}
}
