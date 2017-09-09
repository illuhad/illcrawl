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
#include <memory>

#include "multi_array.hpp"
#include "work_partitioner.hpp"

namespace illcrawl {
namespace util{

/// Translates a C++ data type into fitsio datatypes
template<typename Scalar_type>
struct fits_datatype
{};

template<>
struct fits_datatype<unsigned long long>
{
  static int image_type()
  { return LONGLONG_IMG; }

  static int datatype()
  { return TLONGLONG; }
};

template<>
struct fits_datatype<long long>
{
  static int image_type()
  { return LONGLONG_IMG; }

  static int datatype()
  { return TLONGLONG; }
};

template<>
struct fits_datatype<unsigned>
{
  static int image_type()
  { return LONG_IMG; }

  static int datatype()
  { return TUINT; }
};

template<>
struct fits_datatype<long>
{
  static int image_type()
  { return LONG_IMG; }

  static int datatype()
  { return TLONG; }
};

template<>
struct fits_datatype<int>
{
  static int image_type()
  { return LONG_IMG; }

  static int datatype()
  { return TINT; }
};

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

template<>
struct fits_datatype<const char*>
{
  static int datatype()
  { return TSTRING; }
};

template<>
struct fits_datatype<char*>
{
  static int datatype()
  { return TSTRING; }
};

template<std::size_t N>
struct fits_datatype<char [N]>
{
  static int datatype()
  { return fits_datatype<char*>::datatype(); }
};

static std::string fits_error(int error_code)
{
  char descr[30];
  fits_get_errstatus(error_code, descr);
  std::string result;

  for(int i = 0; i < 30 && descr[i]; ++i)
    result += descr[i];
  return result;
}

class fits_header
{
public:
  fits_header(const std::string& filename)
    : _filename{filename}, _file{nullptr}
  {
    int status = 0;
    if(fits_open_file(&_file, _filename.c_str(), READWRITE, &status))
      throw std::runtime_error{"fits_header: Could not open fits file "+filename};
  }

  ~fits_header()
  {
    int status = 0;
    if(_file != nullptr)
      fits_close_file(_file, &status);
  }

  template<class T>
  void add_entry(const std::string& name,
                 const T& value,
                 const std::string& comment = "")
  {
    assert(_file != nullptr);

    int status = 0;


    fits_write_comment(_file, const_cast<char*>(comment.c_str()), &status);

    if(fits_write_key(_file,
                       fits_datatype<T>::datatype(),
                       name.c_str(),
                       const_cast<T*>(&value),
                       nullptr, &status))
      throw std::runtime_error("Error while updating fits key: "+fits_error(status));
  }

  void add_entry(const std::string& name,
                 const std::string& value,
                 const std::string& comment = "")
  {
    assert(_file != nullptr);

    const char* val = value.c_str();
    int status = 0;
    fits_write_comment(_file, const_cast<char*>(comment.c_str()), &status);

    if(fits_write_key(_file,
                       fits_datatype<char*>::datatype(),
                       name.c_str(),
                       const_cast<char*>(val),
                       comment.c_str(), &status))
      throw std::runtime_error("Error while updating fits key: "+fits_error(status));
  }

private:
  std::string _filename;
  fitsfile* _file;
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
  /// \param out An array that will be used to store the loaded data. It will be
  /// automatically resized to the correct size and dimensions.
  /// \throws std::runtime_error if the data could not be loaded
  void load(util::multi_array<T>& out) const
  {
    fitsfile* file;
    int status = 0;
    int bitpix, naxis_flag;

    if(!fits_open_file(&file, _filename.c_str(), READONLY, &status))
    {
      int dimension = 0;

      if(fits_get_img_dim(file, &dimension, &status))
        throw std::runtime_error("Could not determine dimension of fits image");

      std::vector<long> naxes(dimension, 0);

      if(!fits_get_img_param(file, dimension, &bitpix, &naxis_flag, naxes.data(),
                             &status))
      {
        std::vector<std::size_t> array_sizes;
        array_sizes.reserve(dimension);
        for(std::size_t i = 0; i < naxes.size(); ++i)
          array_sizes.push_back(static_cast<std::size_t>(naxes[i]));

        out = util::multi_array<T>(array_sizes);

        long fpixel [dimension];
        for(std::size_t i = 0; i < static_cast<std::size_t>(dimension); ++i)
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


#ifndef FITS_WITHOUT_MPI
#include <boost/mpi.hpp>
#include "work_partitioner.hpp"

namespace illcrawl{
namespace util {


template<class T>
class distributed_fits_slices
{
public:
  distributed_fits_slices(const work_partitioner& partitioning,
                          const std::string& filename)
    : _comm{partitioning.get_communicator()},
      _partitioning{std::move(partitioning.clone())},
      _filename{filename}
  {}

  void load(util::multi_array<T>& out) const
  {
    fitsfile* file;
    int status = 0;
    int bitpix, naxis_flag;
    std::vector<long> naxes(3, 0);

    if(!fits_open_file(&file, _filename.c_str(), READONLY, &status))
    {
      if(!fits_get_img_param(file, 3, &bitpix, &naxis_flag, naxes.data(),
                             &status))
      {
        std::vector<std::size_t> array_sizes;
        array_sizes.reserve(3);
        for(std::size_t i = 0; i < naxes.size(); ++i)
          array_sizes.push_back(static_cast<std::size_t>(naxes[i]));

        _partitioning->run(array_sizes[2]);
        out = util::multi_array<T>(array_sizes[0], array_sizes[1], _partitioning->get_num_local_jobs());

        long fpixel [3] = {1, 1, _partitioning->own_begin() + 1};
        long lpixel [3] = {array_sizes[0], array_sizes[1], _partitioning->own_end()};
        long inc [3] = {1, 1, 1};

        fits_read_subset(file, fits_datatype<T>::datatype(),
                         fpixel,
                         lpixel,
                         inc,
                         nullptr, out.data(), nullptr, &status);

      }

      fits_close_file(file, &status);
    }
    else
      throw std::runtime_error(std::string("Could not load fits file: ") + _filename);
  }

  void save(const util::multi_array<T>& data) const
  {
    fitsfile* file;
    int status = 0;


    // Create new file on the master process
    if(_comm.rank() == 0)
    {
      long naxes [3] = {
        static_cast<long>(data.get_extent_of_dimension(0)),
        static_cast<long>(data.get_extent_of_dimension(1)),
        static_cast<long>(_partitioning->get_num_global_jobs())
      };

      // cfitsio will only overwrite files when their names are preceded by an
      // exclamation mark...
      std::string fitsio_filename = "!"+_filename;
      if (fits_create_file(&file, fitsio_filename.c_str(), &status))
        throw std::runtime_error(std::string("Could not create fits file: ")
                                 + _filename
                                 + " (Fits Error: "
                                 + fits_error(status)
                                 + ")");

      if (fits_create_img(file, fits_datatype<T>::image_type(),
                           3, naxes, &status))
        throw std::runtime_error(std::string("Could not create fits image: ") + _filename
                                 + " (Fits Error: "
                                 + fits_error(status)
                                 + ")");

      fits_close_file(file, &status);
    }

    // Wait until file is created
    _comm.barrier();

    // Open file on all processes
    if(fits_open_file(&file, _filename.c_str(), READWRITE, &status))
      throw std::runtime_error(std::string("Could not load fits file: ") + _filename);

    long fpixel [] = {1, 1, static_cast<long>(_partitioning->own_begin()) + 1};
    long lpixel [] = {
      static_cast<long>(data.get_extent_of_dimension(0)),
      static_cast<long>(data.get_extent_of_dimension(1)),
      static_cast<long>(_partitioning->own_end())
    };

    fits_write_subset(file,
                      fits_datatype<T>::datatype(),
                      fpixel, lpixel,
                      const_cast<T*>(data.data()),
                      &status);

    fits_close_file(file, &status);
  }

private:
  boost::mpi::communicator _comm;
  std::unique_ptr<work_partitioner> _partitioning;
  std::string _filename;
};

} // util
} // illcrawl



#endif // FITS_WITHOUT_MPI

#endif // FITS_HPP

