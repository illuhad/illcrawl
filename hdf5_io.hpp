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


#ifndef HDF5_IO_HPP
#define HDF5_IO_HPP

#include <vector>
#include <cassert>
#include <type_traits>
#include <H5Cpp.h>
#include <array>
#include <algorithm>
#include <map>

#include "multi_array.hpp"
#include "io_iterator.hpp"

namespace illcrawl{
namespace io {

class illustris_data_loader
{
public:
  illustris_data_loader(const std::string& hdf5_filename, unsigned particle_type_id)
    : _filename{hdf5_filename}, _file{hdf5_filename, H5F_ACC_RDONLY}
  {
    select_group(particle_type_id);
  }

  ~illustris_data_loader()
  {
    _file.close();
  }

  illustris_data_loader& operator=(const illustris_data_loader& other) = delete;
  illustris_data_loader(const illustris_data_loader&) = delete;



  inline
  H5::DataSet get_dataset(const std::string& name) const
  {
    return _group.openDataSet(name);
  }

  static std::string get_coordinate_identifier() {return "Coordinates";}
  static std::string get_density_identifier() {return "Density";}
  static std::string get_internal_energy_identifier() {return "InternalEnergy";}
  static std::string get_smoothing_length_identifier() {return "SmoothingLength";}
  static std::string get_volume_identifier() {return "Volume";}
  static std::string get_masses_identifier() {return "Masses";}
  static std::string get_electron_abundance_identifier() {return "ElectronAbundance";}

private:
  void select_group(unsigned particle_type_id)
  {
    try {
      _group.close();
    } catch (...) {}

    _group = _file.openGroup(_particle_type + std::to_string(particle_type_id));
  }

  const std::string _particle_type = "PartType";

  const std::string _filename;
  H5::H5File _file;
  H5::Group _group;
};

class illustris_gas_data_loader : public illustris_data_loader
{
public:
  illustris_gas_data_loader(const std::string& filename)
    : illustris_data_loader{filename, 0}
  {}


  inline
  H5::DataSet get_coordinates() const
  {
    return get_dataset("Coordinates");
  }

  inline
  H5::DataSet get_density() const
  {
    return get_dataset("Density");
  }

  inline
  H5::DataSet get_internal_energy() const
  {
    return get_dataset("InternalEnergy");
  }

  inline
  H5::DataSet get_smoothing_length() const
  {
    return get_dataset("SmoothingLength");
  }

  inline
  H5::DataSet get_volume() const
  {
    return get_dataset("Volume");
  }
};




template<class T, class Enable = void>
struct hdf5_data_type
{

};

template<class T>
struct hdf5_data_type<T, typename std::enable_if<std::is_integral<T>::value>>
{
  static
  H5::DataType value(const H5::DataSet& data)
  {
    return data.getIntType();
  }
};

template<class T>
struct hdf5_data_type<T, typename std::enable_if<std::is_floating_point<T>::value>>
{
  static
  H5::DataType value(const H5::DataSet& data)
  {
    return data.getFloatType();
  }
};

template<class T> struct hdf5_pred_data_type {};

#define DEFINE_HDF5_PRED_DATA_TYPE(CppType, Hdf5Type) \
  template<> struct hdf5_pred_data_type<CppType>{ static H5::DataType value(){return Hdf5Type; } };

DEFINE_HDF5_PRED_DATA_TYPE(char, H5::PredType::NATIVE_CHAR);
DEFINE_HDF5_PRED_DATA_TYPE(unsigned char, H5::PredType::NATIVE_UCHAR);
DEFINE_HDF5_PRED_DATA_TYPE(short, H5::PredType::NATIVE_SHORT);
DEFINE_HDF5_PRED_DATA_TYPE(unsigned short, H5::PredType::NATIVE_USHORT);
DEFINE_HDF5_PRED_DATA_TYPE(int, H5::PredType::NATIVE_INT);
DEFINE_HDF5_PRED_DATA_TYPE(unsigned, H5::PredType::NATIVE_UINT);
DEFINE_HDF5_PRED_DATA_TYPE(long, H5::PredType::NATIVE_LONG);
DEFINE_HDF5_PRED_DATA_TYPE(unsigned long, H5::PredType::NATIVE_ULONG);
DEFINE_HDF5_PRED_DATA_TYPE(long long, H5::PredType::NATIVE_LLONG);
DEFINE_HDF5_PRED_DATA_TYPE(unsigned long long, H5::PredType::NATIVE_ULLONG);

DEFINE_HDF5_PRED_DATA_TYPE(float, H5::PredType::NATIVE_FLOAT);
DEFINE_HDF5_PRED_DATA_TYPE(double, H5::PredType::NATIVE_DOUBLE);
DEFINE_HDF5_PRED_DATA_TYPE(long double, H5::PredType::NATIVE_LDOUBLE);

template<class T>
class hdf5_writer
{
public:
  hdf5_writer(const std::string& filename)
    : _filename{filename}, _file{filename, H5F_ACC_TRUNC}
  {}

  void add_group(const std::string& group_name, std::size_t size_hint = 0)
  {
    _groups[group_name] = _file.createGroup(group_name, size_hint);
  }

  H5::Group get_group(const std::string& group_name) const
  {
    return _groups.at(group_name);
  }

  void ensure_group_exists(const std::string& group_name, std::size_t size_hint = 0)
  {
    if(_groups.find(group_name) == _groups.end())
      add_group(group_name, size_hint);
  }

  H5::DataSet create_dataset(const std::string& group_name,
                             const std::string& dataset_name,
                             const std::vector<hsize_t>& shape)
  {
    ensure_group_exists(group_name);

    assert(shape.size() > 0);

    H5::DataSpace dataspace{
      static_cast<int>(shape.size()),
      shape.data()
    };


    H5::DataSet dataset = _groups[group_name].createDataSet(
          dataset_name,
          hdf5_pred_data_type<T>::value(),
          dataspace);

    return dataset;
  }

  void add_dataset(const std::string& group_name,
                   const std::string& dataset_name,
                   const std::vector<T>& data)
  {
    std::vector<hsize_t> shape = {static_cast<hsize_t>(data.size())};
    H5::DataSet dset = create_dataset(group_name, dataset_name, shape);

    if(data.size())
      dset.write(data.data(), hdf5_pred_data_type<T>::value());
  }

  void add_dataset(const std::string& group_name,
                   const std::string& dataset_name,
                   const std::vector<hsize_t>& shape,
                   const std::vector<T>& data)
  {
    std::cout << "shape: ";
    for(std::size_t i = 0; i < shape.size(); ++i)
      std::cout << shape[i] << " ";
    std::cout << std::endl;

    std::cout << "data length " << data.size() << std::endl;

    H5::DataSet dset = create_dataset(group_name, dataset_name, shape);

    if(data.size())
      dset.write(data.data(), hdf5_pred_data_type<T>::value());
  }

  void add_dataset(const std::string& group_name,
                   const std::string& dataset_name,
                   const util::multi_array<T>& data)
  {
    std::vector<hsize_t> shape;
    for(std::size_t i = 0; i < data.get_dimension(); ++i)
      shape.push_back(static_cast<hsize_t>(data.get_extent_of_dimension(i)));

    H5::DataSet dset = create_dataset(group_name, dataset_name, shape);

    if(data.size())
      dset.write(data.data(), hdf5_pred_data_type<T>::value());
  }

private:
  std::string _filename;

  H5::H5File _file;

  std::map<std::string, H5::Group> _groups;
};


template<class Data_type>
class buffer_accessor
{
public:
  buffer_accessor(std::size_t total_row_size,
                  const std::vector<std::vector<hsize_t>>& extents)
    : _dataset_id{0}, _total_row_size{total_row_size},
      _extents{extents}, _row_size_subsums(1,0)
  {

    for(std::size_t i = 1; i < _extents.size(); ++i)
    {
      _row_size_subsums.push_back(_row_size_subsums[i-1] + get_dataset_row_size(i-1));
    }
  }

  void select_dataset(std::size_t dataset_id)
  {
    assert(dataset_id < _extents.size());
    _dataset_id = dataset_id;
  }

  std::size_t get_index(std::size_t num_rows_read,
                        std::size_t row, std::size_t col) const
  {
    std::size_t offset = _row_size_subsums[_dataset_id] * num_rows_read;

    return offset + row * get_dataset_row_size(_dataset_id) + col;
  }

  const Data_type& operator()(const std::vector<Data_type>& buffer,
                        std::size_t num_rows_read,
                        std::size_t row, std::size_t col = 0) const
  {
    std::size_t index = get_index(num_rows_read, row, col);
    return buffer[index];
  }

  template<class Dataset_iterator>
  const Data_type& operator()(const Dataset_iterator& iter,
                              std::size_t row, std::size_t col = 0) const
  {
    return (*this)(*iter, iter.get_num_available_rows(), row, col);
  }

  inline
  std::size_t get_dataset_row_size(std::size_t dataset_id) const
  {
    std::size_t row_size = 1;
    if(_extents[dataset_id].size() > 1)
      row_size = _extents[dataset_id][row_size];

    return row_size;
  }

  std::size_t get_num_datasets() const
  {
    return _extents.size();
  }

  const std::vector<hsize_t>& get_dataset_shape(std::size_t dataset_id) const
  {
    assert(dataset_id < _extents.size());
    return _extents[dataset_id];
  }

private:
  std::size_t _dataset_id;

  std::size_t _total_row_size;
  std::vector<std::vector<hsize_t>> _extents;



  std::vector<std::size_t> _row_size_subsums;
};


namespace detail {

template<class Data_type>
class basic_dataset_streamer
{
public:
  using value_type = Data_type;

  explicit basic_dataset_streamer(const std::vector<H5::DataSet>& data)
    : _read_position{0}, _num_rows{0}, _data{data}
  {
    assert(_data.size() > 0);

    for(std::size_t i = 0; i < _data.size(); ++i)
    {
      H5::DataSpace dspace = _data[i].getSpace();
      int rank = dspace.getSimpleExtentNdims();
      assert(rank > 0);

      _extent.push_back(std::vector<hsize_t>(static_cast<std::size_t>(rank), 0));

      dspace.getSimpleExtentDims(_extent.back().data());

      if(_extent.back().size() != 0)
      {
        if(i != 0)
          // All datasets should have the same number of rows
          assert(_num_rows == _extent[i][0]);

        _num_rows = _extent[i][0];
      }
    }
  }

  virtual ~basic_dataset_streamer(){}

  inline
  std::vector<hsize_t> get_extent(int data_set = 0) const
  {
    return _extent[data_set];
  }

  inline
  std::size_t get_num_rows() const
  {
    return _num_rows;
  }

  /// \return The total number of data elements
  /// per row, summing up across all datasets
  inline
  std::size_t get_total_row_size() const
  {
    std::size_t complete_row_size = 0;
    for(std::size_t i = 0; i < _extent.size(); ++i)
    {
      if(_extent[i].size() > 1)
        complete_row_size += _extent[i][1];
      else
        complete_row_size += 1;
    }
    return complete_row_size;
  }

  std::size_t read_list(std::size_t pos, std::size_t n,
                        std::vector<Data_type>& out) const
  {
    return read_table2d(pos, n, out);
  }


  std::size_t read_list(std::size_t n,
                        std::vector<Data_type>& out) const
  {
    return read_table2d(n, out);
  }


  std::size_t read_table2d(std::size_t pos, std::size_t n,
                  std::vector<Data_type>& out) const
  {
    _read_position = pos;

    return read_table2d(n, out);
  }


  std::size_t read_table2d(std::size_t n,
                  std::vector<Data_type>& out) const
  {
    std::fill(out.begin(), out.end(), Data_type());

    // Calculate complete row size across all datasets
    std::size_t complete_row_size = get_total_row_size();


    if (out.size() != complete_row_size * n)
      out.resize(complete_row_size * n);

    hsize_t num_rows_to_read = std::min(n, _num_rows - _read_position);

    if (num_rows_to_read == 0 || (_read_position > _num_rows))
      return 0;

    read(_read_position, num_rows_to_read, out.data());

    _read_position += num_rows_to_read;
    // return the number of rows that have been read
    return num_rows_to_read;
  }

  inline
  std::size_t get_read_position() const
  {
    return _read_position;
  }

  buffer_accessor<Data_type> create_buffer_accessor() const
  {
    return buffer_accessor<Data_type>{get_total_row_size(), _extent};
  }

protected:

  void read_hdf5_data_block(std::size_t position,
                            std::size_t num_rows_to_read,
                       Data_type* out) const
  {
    std::size_t output_offset = 0;

    for (std::size_t i = 0; i < _extent.size(); ++i)
    {
      std::size_t row_size = 1;
      if (_extent[i].size() > 1)
        row_size = _extent[i][1];

      H5::DataSpace dspace = _data[i].getSpace();

      hsize_t count[2] = {num_rows_to_read, row_size};
      hsize_t offset[2] = {position, 0};
      dspace.selectHyperslab(H5S_SELECT_SET, count, offset);

      hsize_t mem_dimension[2] = {num_rows_to_read, row_size};

      H5::DataSpace memory_space{static_cast<int>(_extent[i].size()),
                                 mem_dimension};
      hsize_t offset_out[2] = {0, 0};
      memory_space.selectHyperslab(H5S_SELECT_SET, count, offset_out);

      _data[i].read(out + output_offset,
           hdf5_pred_data_type<Data_type>::value(), memory_space, dspace);

      output_offset += num_rows_to_read * row_size;
    }
  }

  virtual void read(std::size_t position,
                    std::size_t num_rows_to_read,
                    Data_type* out) const
  {
    read_hdf5_data_block(position, num_rows_to_read, out);
  }

private:

  std::vector<std::vector<hsize_t>> _extent;

  mutable std::size_t _read_position;
  std::size_t _num_rows;

  std::vector<H5::DataSet> _data;
};

}

template<class Data_type>
class sync_dataset_streamer : public detail::basic_dataset_streamer<Data_type>
{
public:
  using const_iterator = detail::dataset_block_iterator<sync_dataset_streamer<Data_type>>;

  sync_dataset_streamer(const std::vector<H5::DataSet>& data)
    : detail::basic_dataset_streamer<Data_type> {data}
  {}

  virtual ~sync_dataset_streamer() {}

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
};

template<class Data_type>
using default_dataset_streamer = sync_dataset_streamer<Data_type>;


}
}

#endif // HDF5_IO_HPP
