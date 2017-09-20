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

#include "dm_reconstruction_backend_grid.hpp"
#include "projective_smoothing_backend.hpp"

namespace illcrawl {
namespace reconstruction_backends {
namespace dm {

class projective_smoothing_grid : public projective_smoothing_backend,
                                  public grid
{
public:
  projective_smoothing_grid(const qcl::device_context_ptr& ctx,
                            const H5::DataSet& smoothing_lengths);

  virtual ~projective_smoothing_grid(){}

  virtual void setup_projected_particles(
                   const cl::Buffer& particles,
                   std::size_t num_particles,
                   const std::vector<cl::Buffer>& additional_datasets) override;
private:
  qcl::device_context_ptr _ctx;
};


}
}
}
