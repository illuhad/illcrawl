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


#ifndef TABULATED_FUNCTION
#define TABULATED_FUNCTION

#include <vector>
#include <cassert>
#include <algorithm>

#include "qcl.hpp"
#include "cl_types.hpp"

namespace illcrawl {
namespace util {

class tabulated_function
{
public:

  template<class Input_iterator>
  tabulated_function(const qcl::device_context_ptr& ctx,
                     Input_iterator y_begin,
                     Input_iterator y_end,
                     device_scalar x_start,
                     device_scalar dx)
    : _ctx{ctx}, _dx{dx}, _x_min{0}
  {
    assert(dx > 0);
    assert(_ctx != nullptr);

    if(y_begin == y_end)
      return;

    _x_min = x_start;

    _y = std::vector<device_scalar>(y_end - y_begin);
    std::copy(y_begin, y_end, _y.begin());

    init();
  }

  const cl::Image1D& get_tabulated_function_values() const
  {
    return _y_buffer;
  }

  std::size_t get_num_function_values() const
  {
    return _y.size();
  }

  device_scalar get_min_x() const
  {
    return _x_min;
  }

  device_scalar get_dx() const
  {
    return _dx;
  }

private:
  void init()
  {
    cl_int err;
    _y_buffer = cl::Image1D{
        _ctx->get_context(),
        CL_MEM_READ_ONLY,
        cl::ImageFormat{CL_R, CL_FLOAT},
        _y.size(),
        _y.data(),
        &err
    };
    _ctx->get_command_queue().enqueueWriteImage(_y_buffer, CL_TRUE, {{0,0,0}}, {{_y.size(),1,1}},0,0,_y.data());
    qcl::check_cl_error(err, "Could not create 1D image for the tabulated function");
  }

  qcl::device_context_ptr _ctx;

  device_scalar _dx;
  device_scalar _x_min;

  std::vector<device_scalar> _y;

  cl::Image1D _y_buffer;
};

/// Base class for 2D tabulated functions
class tabulated_function2d
{
public:
  template<class Input_iterator>
  tabulated_function2d(const qcl::device_context_ptr& ctx,
                       Input_iterator data_begin,
                       Input_iterator data_end,
                       const device_vector2& xy_start,
                       const device_vector2& delta,
                       std::size_t row_size,
                       std::size_t column_size)
    : _ctx{ctx},
      _xy_start{xy_start},
      _delta{delta},
      _x_size{row_size},
      _y_size{column_size}
  {
    init(data_begin, data_end);
  }

  tabulated_function2d(const qcl::device_context_ptr& ctx,
                       const device_vector2& xy_start,
                       const device_vector2& delta)
    : _ctx{ctx},
      _xy_start{xy_start},
      _delta{delta},
      _x_size{0},
      _y_size{0}
  {}


  const cl::Image2D& get_tabulated_function_values() const
  {
    return _table_buffer;
  }

  std::size_t get_num_function_values() const
  {
    return _data.size();
  }

  std::size_t get_x_size() const
  {
    return _x_size;
  }

  std::size_t get_y_size() const
  {
    return _y_size;
  }

  device_vector2 get_min_xy() const
  {
    return _xy_start;
  }

  device_vector2 get_delta() const
  {
    return _delta;
  }

protected:
  template<class Input_iterator>
  void init(std::size_t row_size,
            std::size_t column_size,
            Input_iterator data_begin,
            Input_iterator data_end)
  {
    this->_x_size = row_size;
    this->_y_size = column_size;
    this->init(data_begin, data_end);
  }

  template<class Input_iterator>
  void init(Input_iterator data_begin,
            Input_iterator data_end)
  {
    assert(_ctx != nullptr);
    assert(_delta.s[0] > 0.0f);
    assert(_delta.s[1] > 0.0f);

    _data = std::vector<device_scalar>(data_end - data_begin);
    assert(_data.size() == _x_size * _y_size);

    std::copy(data_begin, data_end, _data.begin());

    cl_int err;
    _table_buffer = cl::Image2D{
        _ctx->get_context(),
        CL_MEM_READ_ONLY,
        cl::ImageFormat{CL_R, CL_FLOAT},
        _x_size,
        _y_size,
        0,
        _data.data(),
        &err
    };

    qcl::check_cl_error(err, "Could not create 2D image for the tabulated function");
    _ctx->get_command_queue().enqueueWriteImage(_table_buffer, CL_TRUE,
                                               {{0,0,0}},
                                               {{_x_size,_y_size,1}},
                                               0,0,_data.data());

  }

private:
  qcl::device_context_ptr _ctx;

  device_vector2 _xy_start;
  device_vector2 _delta;

  std::size_t _x_size;
  std::size_t _y_size;

  std::vector<device_scalar> _data;

  cl::Image2D _table_buffer;
};


}
}

#endif
