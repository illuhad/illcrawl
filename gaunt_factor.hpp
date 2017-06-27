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

#ifndef GAUNT_FACTOR
#define GAUNT_FACTOR

#include "tabulated_function.hpp"
#include "qcl.hpp"

namespace illcrawl {
namespace model {
namespace gaunt {

/// free-free thermally averaged gaunt factor
/// according to the values by van Hoof (2014)
class thermally_averaged_ff : public util::tabulated_function2d
{
public:
  thermally_averaged_ff(const qcl::device_context_ptr& ctx);
};

} // gaunt
} // model
} // illcrawl

#endif
