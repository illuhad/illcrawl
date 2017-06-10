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

#ifndef CHANDRA_HPP
#define CHANDRA_HPP

#include "tabulated_function.hpp"

namespace illcrawl {
namespace chandra {

/// Cycle 19 chandra area response function
class arf : public util::tabulated_function{
public:
  /// \param ctx The OpenCL context
  arf(const qcl::device_context_ptr& ctx);
  virtual ~arf(){}
};

}
}

#endif

