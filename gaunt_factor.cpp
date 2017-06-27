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

#include "gaunt_factor.hpp"
#include "multi_array.hpp"
#include "data_table_loader.hpp"

namespace illcrawl {
namespace model {
namespace gaunt {


thermally_averaged_ff::thermally_averaged_ff(const qcl::device_context_ptr& ctx)
  : tabulated_function2d{ctx, device_vector2{-6.f, -16.f}, device_vector2{0.2f, 0.2f}}
{
  util::data_table_loader<device_scalar> table_loader{"gaunt_factor.dat"};

  const util::multi_array<device_scalar>& table = table_loader();

  assert(table.get_dimension() == 2);
  this->init(table.get_extent_of_dimension(0),
             table.get_extent_of_dimension(1),
             table.begin(), table.end());
}

} // gaunt
} // model
} // illcrawl
