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

#ifndef MULTI_ARRAY_HPP
#define MULTI_ARRAY_HPP

#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#ifdef MULTI_ARRAY_WITH_MPI
#include <boost/mpi.hpp>
#define MULTI_ARRAY_WITH_SERIALIZATION
#endif
#ifdef MULTI_ARRAY_WITH_SERIALIZATION
#include <boost/serialization/vector.hpp>
#endif

#if __cplusplus >= 201103L
#define NULLPTR nullptr
#else
#define NULLPTR NULL
#endif

namespace illcrawl {
namespace util {


/// Implements a dynamic multidimensional array
/// \tparam T the data type to be stored in the array
template<typename T>
class multi_array
{
public:
  typedef std::size_t size_type;
  typedef std::size_t index_type;

  typedef T* iterator;
  typedef const T* const_iterator;
  
  /// Construct empty array with no dimensions

  multi_array()
  : buffer_size_(0), data_(NULLPTR)
  {
  }

  /// Construct multi dimensional array with the dimensions given
  /// as a \c std::vector.
  /// \param sizes Specifies the dimensions. \c sizes.size() is the number
  /// of dimensions, and \c sizes[i] the extent in the i-th dimension.
  /// E.g, to construct a 2x3 array, \c sizes has to contain the elements {2, 3}.

  explicit multi_array(const std::vector<size_type>& sizes)
  : sizes_(sizes), data_(NULLPTR)
  {
    init();
  }

  /// Construct multi dimensional array with the dimensions given
  /// as a simple stack-based C-style array.
  /// \param sizes Specifies the dimensions. \c sizes.size() is the number
  /// of dimensions, and \c sizes[i] the extent in the i-th dimension.
  /// E.g, to construct a 2x3 array, \c sizes has to contain the elements {2, 3}.

  template<size_type N>
  explicit multi_array(const size_type(&sizes) [N])
  : sizes_(sizes + 0, sizes + N), data_(NULLPTR)
  {
    init();
  }

  /// Construct multi dimensional array with the dimensions given
  /// as a simple C-style array.
  /// \param sizes Specifies the dimensions. \c sizes.size() is the number
  /// of dimensions, and \c sizes[i] the extent in the i-th dimension.
  /// E.g, to construct a 2x3 array, \c sizes has to contain the elements {2, 3}.
  /// \param num_dimensions The number of elements of \c sizes and thus
  /// the number of dimensions of the \c multi_array

  multi_array(const size_type* sizes, size_type num_dimensions)
  : sizes_(sizes, sizes + num_dimensions), data_(NULLPTR)
  {
    init();
  }

  /// Construct two dimensional array
  /// \param size_x The extent of the array in dimension 0
  /// \param size_y The extent of the array in dimension 1

  multi_array(size_type size_x, size_type size_y)
  : data_(NULLPTR)
  {
    sizes_.reserve(2);
    sizes_.push_back(size_x);
    sizes_.push_back(size_y);
    init();
  }

  /// Construct three dimensional array
  /// \param size_x The extent of the array in dimension 0
  /// \param size_y The extent of the array in dimension 1
  /// \param size_z The extent of the array in dimension 2

  multi_array(size_type size_x, size_type size_y, size_type size_z)
  : data_(NULLPTR)
  {
    sizes_.reserve(3);
    sizes_.push_back(size_x);
    sizes_.push_back(size_y);
    sizes_.push_back(size_z);
    init();
  }

  /// Copy Constructor. May Throw.
  /// \param other The other instance of which the content shall be copied
  multi_array(const multi_array<T>& other)
  : sizes_(other.sizes_), position_increments_(other.position_increments_),
    buffer_size_(other.buffer_size_), data_(NULLPTR)
  {
    if (sizes_.size() != 0)
    {
      init();
      std::copy(other.begin(), other.end(), data_);
    }
  }

  /// Assignment operator. Provides strong exception guarantee.

  multi_array<T>& operator=(multi_array<T> other)
  {
    // using the copy and swap idiom (note the by-value function
    // parameter!) grants us a strong exception guarantee
    swap(*this, other);
    return *this;
  }

  ~multi_array()
  {
    if (data_)
      delete [] data_;
  }

  /// Swap two multi arrays. Their sizes do not have to equal.
  /// \param a The first array
  /// \param b The second array

  friend void swap(multi_array<T>& a, multi_array<T>& b)
  {
    using std::swap;

    swap(a.sizes_, b.sizes_);
    swap(a.buffer_size_, b.buffer_size_);
    swap(a.data_, b.data_);
    swap(a.position_increments_, b.position_increments_);
  }

  /// \return The extent of a dimension
  /// \param dim The index of the dimension

  size_type get_extent_of_dimension(std::size_t dim) const
  {
    assert(dim < sizes_.size());
    return sizes_[dim];
  }

  /// \return The dimension of the array

  size_type get_dimension() const
  {
    return sizes_.size();
  }

  /// \return An iterator type to the beginning of the array

  iterator begin()
  {
    return data_;
  }

  /// \return An iterator type to the beginning of the array

  const_iterator begin() const
  {
    return data_;
  }

  /// \return An iterator type pointing to one element beyond the
  /// last element of the array

  iterator end()
  {
    return data_ + buffer_size_;
  }

  /// \return An iterator type pointing to one element beyond the
  /// last element of the array

  const_iterator end() const
  {
    return data_ + buffer_size_;
  }

  /// \return The total number of elements in the array

  size_type get_num_elements() const
  {
    return buffer_size_;
  }

  /// \return The total number of elements in the array

  size_type size() const
  {
    return get_num_elements();
  }

  /// Grants access to the raw data buffer
  /// \return the raw data buffer
  T* data()
  {
    return data_;
  }

  /// Grants access to the raw data buffer
  /// \return the raw data buffer
  const T* data() const
  {
    return data_;
  }

  /// Checks if an (multi-dimensional) index is within the bounds
  /// of the multi_array
  /// \return whether the index is within the bounds
  /// \param position the index
  bool is_within_bounds(const index_type* position) const
  {
    for(size_type i = 0; i < get_dimension(); ++i)
      if(position[i] >= sizes_[i])
        return false;
    return true;
  }

  /// Checks if an (multi-dimensional) index is within the bounds
  // of the multi_array
  /// \return whether the index is within the bounds
  /// \param position the index
  bool is_within_bounds(const std::vector<index_type>& position) const
  {
    for(size_type i = 0; i < get_dimension(); ++i)
      if(position[i] >= sizes_[i])
        return false;
    return true;
  }



  /// Access an element of the array
  /// \param position Contains the indices of the element to look up
  /// \return A reference to the specified element

  T& operator[](const std::vector<index_type>& position)
  {
    assert(position.size() == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position Contains the indices of the element to look up
  /// \return A reference to the specified element

  const T& operator[](const std::vector<index_type>& position) const
  {
    assert(position.size() == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position Contains the indices of the element to look up
  /// \return A reference to the specified element

  T& operator[](const std::vector<int>& position)
  {
    assert(position.size() == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position Contains the indices of the element to look up
  /// \return A reference to the specified element

  const T& operator[](const std::vector<int>& position) const
  {
    assert(position.size() == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position A simple C-array containing the indices of the
  /// element to look up
  /// \return A reference to the specified element

  template<size_type N>
  T& operator[](const index_type(&position) [N])
  {
    assert(N == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position A simple C-array containing the indices of the
  /// element to look up
  /// \return A reference to the specified element

  template<size_type N>
  const T& operator[](const index_type(&position) [N]) const
  {
    assert(N == get_dimension());
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position A pointer to a simple C-array containing the
  /// indices of the element to look up
  /// \return A reference to the specified element

  T& operator[](const index_type* position)
  {
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);

    assert(pos < buffer_size_);
    return data_[pos];
  }

  /// Access an element of the array
  /// \param position A pointer to asimple C-array containing the
  /// indices of the element to look up
  /// \return A reference to the specified element

  const T& operator[](const index_type* position) const
  {
    assert(data_ != NULLPTR);

    size_type pos = calculate_position(position);
    assert(pos < buffer_size_);
    return data_[pos];
  }

#if __cplusplus >= 201103L
  template<std::size_t Dim>
  T& operator[](const std::array<index_type, Dim>& position)
  {
    return (*this)[position.data()];
  }

  template<std::size_t Dim>
  const T& operator[](const std::array<index_type, Dim>& position) const
  {
    return (*this)[position.data()];
  }

#endif

#ifdef MULTI_ARRAY_WITH_MPI
  
#if __cplusplus >= 201103L
  /// Combines each element within all parallel instances of the array on the
  /// master process by applying a user specified combination function. This
  /// function does not rely on \c boost::serialization to send the data, since
  /// this would lead to an additional call to \c MPI_Alloc_mem which fails
  /// if the array is larger than \f$2^{31}-1\f$ bytes.
  /// \param comm The mpi communicator
  /// \param master_rank The rank of the master process. Only this process
  /// will have access to the combined array.
  /// \param combination_method The function that will be applied to combine
  /// all elements at the same position in the parallel array instances.
  /// It must have the signature \c void(T&, const T&) and be an in-place operation

  template<class Function>
  void reduce_parallel_array(const boost::mpi::communicator& comm,
                             int master_rank,
                             Function combination_method)
  {
    int rank = comm.rank();
    int nprocs = comm.size();

    if(rank != master_rank)
    {
      // Use synchronous communication to save memory (the data tables can be quite large)
      comm.send(master_rank, 0, this->sizes_);
      comm.send(master_rank, 0, this->data_, buffer_size_);
    }
    else
    {
      for(int proc = 0; proc < nprocs; ++proc)
      {
        if(proc != master_rank)
        {
          util::multi_array<T> recv_data;
          comm.recv(proc, 0, recv_data.sizes_);
          recv_data.init();
          comm.recv(proc, 0, recv_data.data_, recv_data.buffer_size_);

          assert(recv_data.get_dimension() == this->get_dimension());
          for(std::size_t dim = 0; dim < recv_data.get_dimension(); ++dim)
            assert(recv_data.get_extent_of_dimension(dim) == this->get_extent_of_dimension(dim));

          auto recv_element = recv_data.begin();
          auto data_element = this->begin();

          for(; recv_element != recv_data.end(); ++recv_element, ++data_element)
            combination_method(*data_element, *recv_element);

        }
      }
    }   
  }

  /// Like \c reduce_parallel_array, but makes the result available to each
  /// process.  This function does not rely on \c boost::serialization to send 
  /// the data, since this would lead to an additional call to \c MPI_Alloc_mem 
  /// which fails if the array is larger than \f$2^{31}-1\f$ bytes.
  /// \param comm The mpi communicator
  /// \param combination_method The function that will be applied to combine
  /// all elements at the same position in the parallel array instances.
  /// It must have the signature T (const T&, const T&)
  template<class Function>
  void allreduce_parallel_array(const boost::mpi::communicator& comm,
                             Function combination_method)
  {
    int master_rank = 0;

    reduce_parallel_array(comm, master_rank, combination_method);

    broadcast(comm, master_rank);
  }
#endif
  
  /// Performs a MPI broadcast operation on this array
  /// \param comm The mpi communicator
  /// \param master_rank The rank of the master process from which the data
  /// shall be broadcast.
  void broadcast(const boost::mpi::communicator& comm, int master_rank)
  {
    boost::mpi::broadcast(comm, this->sizes_, master_rank);
    
    if(comm.rank() != master_rank)
      this->init();
    
    boost::mpi::broadcast(comm, this->data_, this->buffer_size_, master_rank);
  }

#endif

private:
  /// Based on the indices of an element, calculates the position
  /// of the element in the flat, one-dimensional data array.
  /// \param position An object of a type offering operator[], that
  /// stores the indices of the element
  /// \return the position in the one dimensional data array.
  template<typename Container>
  inline size_type calculate_position(const Container& position) const
  {
    size_type pos = 0;
    for (size_type i = 0; i < get_dimension(); ++i)
      pos += position[i] * position_increments_[i];

    return pos;
  }

  /// Initializes the data array and the position increments of each dimension
  void init()
  {
    assert(sizes_.size() != 0);
    for (std::size_t i = 0; i < sizes_.size(); ++i)
      assert(sizes_[i] != 0);

    if(data_ != NULLPTR)
      delete [] data_;

    buffer_size_ = std::accumulate(sizes_.begin(), sizes_.end(), 1,
                                   std::multiplies<size_type>());

    data_ = new T [buffer_size_];

    position_increments_.resize(get_dimension());
    position_increments_[0] = 1;

    for (std::size_t i = 1; i < sizes_.size(); ++i)
    {
      position_increments_[i] = position_increments_[i - 1] * sizes_[i - 1];
    }
  }

  std::vector<size_type> sizes_;
  std::vector<index_type> position_increments_;

  size_type buffer_size_;

  T* data_;

#ifdef MULTI_ARRAY_WITH_SERIALIZATION
  friend class boost::serialization::access;

  /// Serializes the array
  template<class Archive>
  void save(Archive& ar, const unsigned int version) const
  {
    ar & sizes_;
    for(size_t i = 0; i < buffer_size_; ++i)
      ar & data_[i];
  }

  /// Deserializes the array
  template<class Archive>
  void load(Archive& ar, const unsigned int version)
  {
    ar & sizes_;
    init();
    for(size_t i = 0; i < buffer_size_; ++i)
      ar & data_[i];
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif
};

}
}

#endif /* MULTI_ARRAY_HPP */

