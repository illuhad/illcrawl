/*
 * This file is part of nanolens, a free program to calculate microlensing 
 * magnification patterns.
 * Copyright (C) 2015  Aksel Alpay
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

#ifndef FITS_HPP
#define	FITS_HPP

#include <fitsio.h>
#include <string>
#include <stdexcept>
#include <vector>

#include "multi_array.hpp"

namespace illcrawl {
namespace util{

/// Translates a C++ data type into fitsio datatypes
template<typename Scalar_type>
struct fits_datatype
{};

template<>
struct fits_datatype<float>
{
  static int image_type()
  { return FLOAT_IMG; }
  
  static int datatype()
  { return TFLOAT; }
};

template<>
struct fits_datatype<double>
{
  static int image_type()
  { return DOUBLE_IMG; }
  
  static int datatype()
  { return TDOUBLE; }
};

/// Implements loading and saving fits images. It is the responsibility of the
/// user to take care of parallelization effects, i.e. preventing two processes 
/// from saving to the same file at the same time or distributing loaded data
/// among computing processes.
/// \tparam T the datatype of the fits image. If the \c T does not equal the actual
/// datatype, cfitsio should automatically convert the data to \c T on the fly.
template<typename T>
class fits
{
public:
  /// Construct object
  /// \param filename The name of the file that shall be loaded or to which
  /// data shall be saved.
  fits(const std::string& filename)
  : _filename(filename) {}
  
  /// Saves data to the file specified in the constructor.
  /// \param data The data that shall be saved.
  /// \throws std::runtime_error if the data could not be saved
  void save(const util::multi_array<T>& data) const
  {
    fitsfile* file;
    int status = 0;

    std::vector<long> naxes;
    
    for(std::size_t dim = 0; dim < data.get_dimension(); ++dim)
      naxes.push_back(data.get_extent_of_dimension(dim));

    // cfitsio will only overwrite files when their names are preceded by an
    // exclamation mark...
    std::string fitsio_filename = "!"+_filename;

    if (!fits_create_file(&file, fitsio_filename.c_str(), &status))
    {
      std::vector<long> fpixel(data.get_dimension(), 1);
      
      if (!fits_create_img(file, fits_datatype<T>::image_type(), 
                           naxes.size(), naxes.data(), &status))
      {
        fits_write_pix(file, fits_datatype<T>::datatype(), fpixel.data(), 
                       data.get_num_elements(), const_cast<T*>(data.data()), &status);
        
        fits_close_file(file, &status);
      }
      else
        throw std::runtime_error(std::string("Could not create fits image: ") + _filename);
      
    }
    else
      throw std::runtime_error(std::string("Could not create fits file: ") + _filename);
  }
  
  /// Loads data from a fits file
  /// \tparam N_dimensions The number of dimensions of the fits image that shall
  /// be loaded
  /// \param out An array that will be used to store the loaded data. It will be
  /// automatically resized to the correct size and dimensions.
  /// \throws std::runtime_error if the data could not be loaded
  template<std::size_t N_dimensions>
  void load(util::multi_array<T>& out) const
  {
#if __cplusplus >= 201103L
    static_assert(N_dimensions > 0, "Number of dimensions cannot be zero.");
#else
    assert(N_dimensions > 0);
#endif
    
    fitsfile* file;
    int status = 0;
    int bitpix, naxis_flag;
    std::vector<long> naxes(N_dimensions, 0);
    
    if(!fits_open_file(&file, _filename.c_str(), READONLY, &status))
    {
      if(!fits_get_img_param(file, N_dimensions, &bitpix, &naxis_flag, naxes.data(), 
                             &status))
      {
        std::vector<std::size_t> array_sizes;
        array_sizes.reserve(N_dimensions);
        for(std::size_t i = 0; i < naxes.size(); ++i)
          array_sizes.push_back(static_cast<std::size_t>(naxes[i]));
        
        out = util::multi_array<T>(array_sizes);
        
        long fpixel [N_dimensions];
        for(std::size_t i = 0; i < N_dimensions; ++i)
          fpixel[i] = 1;
   
        fits_read_pix(file,
                        fits_datatype<T>::datatype(), 
                        fpixel, 
                        out.size(), 
                        NULL, 
                        out.data(), 
                        NULL, 
                        &status);
        
      }
      
      fits_close_file(file, &status);
    }
    else
      throw std::runtime_error(std::string("Could not load fits file: ") + _filename);
  }
  
private:
  std::string _filename;
};

}
}

#endif	/* FITS_HPP */

