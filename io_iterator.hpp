#ifndef IO_ITERATOR_HPP
#define IO_ITERATOR_HPP

#include <vector>
#include <cassert>

namespace illcrawl {
namespace io {
namespace detail {


template<class Streamer_type>
class dataset_block_iterator
{
public:

  using value_type = typename Streamer_type::value_type;
  using buffer_type = std::vector<value_type>;

  using iterator = dataset_block_iterator<Streamer_type>;
  using reference = const buffer_type&;
  using pointer = const value_type*;

  dataset_block_iterator()
    : _streamer{nullptr}, _position{0}, _num_available_rows{0},
      _queued_read_begin{0}, _queued_read_end{0}, _block_size{0}
  {}

  explicit
  dataset_block_iterator(const Streamer_type* streamer, std::size_t pos = 0, std::size_t block_size = 512)
    : _streamer{streamer}, _position{pos}, _num_available_rows{0},
      _queued_read_begin{0}, _queued_read_end{0}, _block_size{block_size}
  {
    assert(_streamer != nullptr);

  }

  dataset_block_iterator(const dataset_block_iterator& other) = default;
  dataset_block_iterator& operator= (const dataset_block_iterator& other) = default;

  virtual ~dataset_block_iterator(){}

  iterator& operator++()
  {
    this->move_forward();
    return *this;
  }

  iterator operator++(int)
  {
    iterator copy{*this};
    this->move_forward();
    return copy;
  }

  inline friend bool operator==(const iterator& a, const iterator& b) noexcept
  {
    if(a.is_end() && b.is_end())
      return true;

    return a._streamer == b._streamer
        && a._position == b._position
        && a._block_size == b._block_size;
  }

  inline friend bool operator!=(const iterator& a, const iterator& b) noexcept
  {
    return !(a == b);
  }

  inline reference operator*() const noexcept
  {
    return get_available_data();
  }

  inline pointer raw_data_buffer() const noexcept
  {
    return get_available_data().data();
  }

  inline pointer operator->() const noexcept
  {
    return get_available_data().data();
  }

  inline const value_type& operator[](std::size_t index) const noexcept
  {
    return get_available_data()[index];
  }

  inline std::size_t get_read_position() const noexcept
  {
    return _position;
  }

  inline bool is_end() const noexcept
  {
    if(_streamer == nullptr)
      return true;

    if(_position >= _streamer->get_num_rows())
      return true;


    return false;
  }

  std::size_t get_num_available_rows() const noexcept
  {
    return _num_available_rows;
  }

  std::size_t get_block_size() const noexcept
  {
    return _block_size;
  }

  std::size_t get_queued_range_begin() const noexcept
  {
    return _queued_read_begin;
  }

  std::size_t get_queued_range_end() const noexcept
  {
    return _queued_read_end;
  }

protected:

  const Streamer_type* get_streamer() const
  {
    return _streamer;
  }

  std::size_t read_data()
  {
    if(_streamer == nullptr)
    {
      return 0;
    }
    _queued_read_begin = _position;
    _queued_read_end = _position + _block_size;

    return _streamer->read_table2d(_position, _block_size, _read_buffer);
  }

  virtual reference get_available_data() const noexcept
  {
    return _read_buffer;
  }

  virtual void move_forward()
  {
    if(is_end())
    {
      _num_available_rows = 0;
      _position += _block_size;
      return;
    }

    _num_available_rows = read_data();

    _position += _block_size;
  }

private:
  const Streamer_type* _streamer;

protected:
  std::vector<value_type> _read_buffer;

  std::size_t _position;
  std::size_t _num_available_rows;

private:
  std::size_t _queued_read_begin;
  std::size_t _queued_read_end;

  std::size_t _block_size;
};

template<class Streamer_type>
class async_dataset_block_iterator : public dataset_block_iterator<Streamer_type>
{
public:
  using value_type = typename Streamer_type::value_type;
  using iterator = async_dataset_block_iterator<Streamer_type>;

  using typename dataset_block_iterator<Streamer_type>::buffer_type;

  using typename dataset_block_iterator<Streamer_type>::reference;
  using typename dataset_block_iterator<Streamer_type>::pointer;


  async_dataset_block_iterator()
    : _num_queued_rows{0},
      _available_range_begin{0},
      _available_range_end{0},
      _operations_pending{false}
  {}
  virtual ~async_dataset_block_iterator() {}

  explicit
  async_dataset_block_iterator(const Streamer_type* streamer,
                               std::size_t pos = 0,
                               std::size_t block_size = 512)
    : dataset_block_iterator<Streamer_type>{streamer, pos, block_size},
      _num_queued_rows{0},
      _available_range_begin{0},
      _available_range_end{0},
      _operations_pending{false}
  {}

  void await_data()
  {

    if(_operations_pending)
    {
      this->get_streamer()->wait();
      std::swap(this->_read_buffer, this->_available_data_buffer);
    }

    if(this->_num_queued_rows == 0)
      this->_num_available_rows = 0;
    else
    {
      this->_num_available_rows = this->_num_queued_rows;
      this->_available_range_begin = this->get_queued_range_begin();
      this->_available_range_end = this->get_queued_range_end();
    }

    this->_operations_pending = false;
  }

  inline bool operations_pending() const
  {
    return _operations_pending;
  }

  std::size_t get_available_data_range_begin() const noexcept
  {
    return _available_range_begin;
  }

  std::size_t get_available_data_range_end() const noexcept
  {
    return _available_range_end;
  }
protected:
  // This causes the base class to return \c _available_data_buffer
  // instead of the read_buffer for operator* and operator->
  virtual reference get_available_data() const noexcept override
  {
    return _available_data_buffer;
  }

  virtual void move_forward() override
  {
    await_data();

    if(this->is_end())
    {
      _num_queued_rows = 0;
      return;
    }

    _num_queued_rows = this->read_data();
    _operations_pending = true;

    this->_position += _num_queued_rows;
  }

private:

  buffer_type _available_data_buffer;
  std::size_t _num_queued_rows;

  std::size_t _available_range_begin;
  std::size_t _available_range_end;

  bool _operations_pending;
};


} // detail
} // io
} // illcrawl

#endif
