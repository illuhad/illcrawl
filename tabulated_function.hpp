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
  template<class Input_iterator1,
           class Input_iterator2>
  tabulated_function(const qcl::device_context_ptr& ctx,
                     Input_iterator1 x_begin,
                     Input_iterator1 x_end,
                     Input_iterator2 y_begin,
                     device_scalar dx)
    : _ctx{ctx}, _dx{dx}, _x_min{0}
  {
    assert(dx > 0);
    assert(_ctx != nullptr);

    if(x_begin == x_end)
      return;

    Input_iterator1 x_iter = x_begin;
    Input_iterator2 y_iter = y_begin;

    std::vector<device_scalar> x;
    std::vector<device_scalar> y;
    for(; x_iter != x_end; ++x_iter, ++y_iter)
    {
      x.push_back(*x_iter);
      y.push_back(*y_iter);
    }

    _x_min = x.front();
    device_scalar x_max = x.back();

    std::size_t current_x_index = 0;

    for(device_scalar x_pos = _x_min; x_pos <= x_max; x_pos += dx)
    {
      while(current_x_index < x.size() &&
            x[current_x_index] > x_pos)
        ++current_x_index;

      if(current_x_index + 1 < x.size())
      {
        // Interpolate
        device_scalar y0 = y[current_x_index];
        device_scalar y1 = y[current_x_index + 1];

        device_scalar x0 = x[current_x_index];
        device_scalar deltax = x[current_x_index + 1] - x0;
        assert(deltax > 0);

        device_scalar rel_pos = (x_pos - x0) / deltax;
        _y.push_back((1 - rel_pos)*y0 + rel_pos*y1);
      }
      else
      {
        _y.push_back(y.back());
      }
    }

    init();

  }

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


}
}

#endif
