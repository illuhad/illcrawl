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


template<class Data_type>
class async_dataset_streamer : public detail::basic_dataset_streamer<Data_type>
{
public:
  async_dataset_streamer(const std::vector<H5::DataSet>& data)
    :  detail::basic_dataset_streamer<Data_type>{data}
  {}

  async_dataset_streamer& operator=(const async_dataset_streamer& other) = delete;
  async_dataset_streamer(const async_dataset_streamer& other) = delete;

  using const_iterator = detail::async_dataset_block_iterator<async_dataset_streamer<Data_type>>;

  virtual ~async_dataset_streamer()
  {}

  void wait() const
  {
    _worker.wait();
  }

  const_iterator begin_row_blocks(std::size_t block_size = 512) const
  {
    auto it = const_iterator{this, 0, block_size};
    ++it; // Make sure the first data block is loaded
    return it;
  }

  const_iterator end_row_blocks() const
  {
    return const_iterator{};
  }
protected:

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
