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


#ifndef ASYNC_IO_HPP
#define ASYNC_IO_HPP


#include "hdf5_io.hpp"
#include "async_worker.hpp"
#include "io_iterator.hpp"

namespace illcrawl {
namespace io {

/// Allows to read data from one or several hdf5 datasets asynchronously.
/// \tparam Data_type the data type of the datasets. Currently, it is assumed
/// that all datasets are of the same type (or can at least all be converted into
/// one type)
template<class Data_type>
class async_dataset_streamer : public detail::basic_dataset_streamer<Data_type>
{
public:
  /// Construct object
  /// \param data The datasets that shall be read. A read operation always reads
  /// the same numbers of 'rows' (i.e. the first dimension, should a dataset be
  /// multidimensional) in one go. It is therefore required that the size of
  /// the first dimension is the same for all datasets.
  async_dataset_streamer(const std::vector<H5::DataSet>& data)
    :  detail::basic_dataset_streamer<Data_type>{data}
  {}

  async_dataset_streamer& operator=(const async_dataset_streamer& other) = delete;
  async_dataset_streamer(const async_dataset_streamer& other) = delete;

  using const_iterator = detail::async_dataset_block_iterator<async_dataset_streamer<Data_type>>;

  virtual ~async_dataset_streamer()
  {}

  /// Wait until pending operations have completed.
  void wait() const
  {
    _worker.wait();
  }

  /// \return an asynchronous iterator to the first block of rows that have
  /// been read.
  /// \param block_size The number of rows that shall be read at once.
  const_iterator begin_row_blocks(std::size_t block_size = 512) const
  {
    auto it = const_iterator{this, 0, block_size};
    ++it; // Make sure the first data block is loaded
    return it;
  }

  /// \return an asynchronous iterator one block beyond
  /// the last row block.
  const_iterator end_row_blocks() const
  {
    return const_iterator{};
  }
protected:

  /// Asynchronously read data rows. Will be called by
  /// the base class.
  /// \param offset where the read shall begin (i.e. the index
  /// of the first row to read)
  /// \param num_elements_to_read how many rows shall be read
  /// \param buf A data buffer to hold the result. This buffer can
  /// only be reused once the \c read() operation is guaranteed
  /// to have completed. This is either the case when \c wait()
  /// is called, or when another asynchronous \c read() operation
  /// starts, because the \c async_worker class used here can only
  /// have one operation pending -- if another operation is queued,
  /// it waits until the first one completes.
  /// The size of \c buf must be at least \c num_elements_to_read * \c row_size
  /// elements, with \c row_size being the sum of the row sizes of
  /// all streamed datasets.
  virtual void read(std::size_t position,
                    std::size_t num_elements_to_read,
                    Data_type* buf) const override
  {
    _worker(
    [position, num_elements_to_read, buf, this]()
    {
      this->read_hdf5_data_block(position,
                                 num_elements_to_read,
                                 buf);
    });
  }

private:
  mutable async::worker_thread _worker;

};


/// Iterate over a data range asynchronously. This allows the process
/// of reading data and the process of processing data to interleave,
/// thereby increasing overall performance. The function is implemented
/// such that while the i-th block is currently processed, the (i+1)-th block
/// is currently being read.
/// \tparam Async_iterator An asynchronous block iterator -- this is a forward
/// iterator with some additional member functions. See \c async_dataset_block_iterator<T>
/// for an example.
/// \param begin An iterator to the first data block of the data range
/// \param end An iterator to one block beyond the last data block of the data range
/// \param f A user supplied function to process a block of data. It must have
/// the signature \c void f(Async_iterator current_block). Note that
/// \c async_for_each_block() already takes care of waiting for asynchronous
/// read operation to complete, when necessary -- it therefore not
/// necessary for the user to explicitly wait for data. However, due to the
/// asynchronous nature, it can happen that the data block is not completely
/// filled when \c f() is called to process it. Therefore, \c f() should always
/// check first how many data rows are actually available using the member
/// functions of the iterator. Particularly, the first call to f() will
/// always be with no available data at all, because in the beginning, no
/// data has been read yet.
template<class Async_iterator, class Function>
void async_for_each_block(Async_iterator begin, Async_iterator end, Function f)
{

  bool processed_all = false;
  bool reached_end = false;
  for(; begin != end || !processed_all; ++begin)
  {
    if(begin == end)
      reached_end = true;

    if(reached_end)
    {
      if(begin.get_available_data_range_end() >= begin.get_queued_range_end())
        processed_all = true;
    }

    f(begin);
  }
}

}
}

#endif
