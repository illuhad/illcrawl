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

#ifndef UNITS_HPP
#define UNITS_HPP

#include "math.hpp"

namespace illcrawl {

/// Converts the standard comoving Illustris
/// units into the units used by illcrawl quantities.
/// In particular, allows for the conversion from comoving
/// quantities to physical ones.
class unit_converter
{
public:
  /// construct object
  /// \param a The scale factor
  /// \param h The hubble constant in units of 100 km/s/Mpc
  unit_converter(math::scalar a = 1.0, math::scalar h = 0.704)
    : _a{a}, _h{h}
  {}

  /// \return conversion factor to turn a quantity in units
  /// of comoving kpc/h into kpc
  math::scalar length_conversion_factor() const
  {
    return _a / _h;
  }

  /// \return factor to convert masses given in
  /// units of 10^10 M_sun/h into units of
  /// M_sun
  math::scalar mass_conversion_factor() const
  {
    return 1.e10 / _h;
  }

  /// \return factor to convert volumes given
  /// in units of (comoving kpc/h)^3 to
  /// kpc^3
  math::scalar volume_conversion_factor() const
  {
    return _a * _a * _a / (_h * _h * _h);
  }

  /// \return factor to convert densities
  /// given in 10^10 M_sun/h / (comoving kpc/h)^3
  /// to M_sun/kpc^3
  math::scalar density_conversion_factor() const
  {
    return mass_conversion_factor() / volume_conversion_factor();
  }

  /// \return factor to convert areas given in
  /// (ckpc/h)^2 to kpc^2
  math::scalar area_conversion_factor() const
  {
    return length_conversion_factor()
         * length_conversion_factor();
  }

  /// \return factor to convert potentials in
  /// km^2/s^2/a to (km/s)^2
  math::scalar potential_conversion_factor() const
  {
    return 1. / _a;
  }

private:
  math::scalar _a;
  math::scalar _h;
};

}

#endif
