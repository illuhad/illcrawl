
#ifndef GRID_HPP
#define GRID_HPP

#include "math.hpp"

namespace illcrawl {
namespace util {

template<std::size_t Dim>
class grid_coordinate_translator
{
public:
  using grid_index = std::array<long long int, Dim>;
  using grid_uindex = std::array<std::size_t, Dim>;

  grid_coordinate_translator()
    : _center{{0.0, 0.0, 0.0}},
      _extent{{1.0,1.0,1.0}},
      _cell_size{{1.0,1.0,1.0}},
      _num_cells{{1,1,1}},
      _min_corner{{-0.5,-0.5,-0.5}}
  {}

  grid_coordinate_translator(const math::vector_n<Dim>& center,
                             const math::vector_n<Dim>& extent,
                             std::size_t num_cells_x)
    : _center{center}, _extent{extent}
  {
    _num_cells[0] = num_cells_x;
    for(std::size_t i = 1; i < Dim; ++i)
      _num_cells[i] = (static_cast<math::scalar>(num_cells_x) / _extent[0]) * _extent[i];

    init();
  }

  grid_coordinate_translator(const math::vector_n<Dim>& center,
                             const math::vector_n<Dim>& extent,
                             const grid_index& num_cells)
    :_center{center}, _extent{extent}
  {
    for(std::size_t i = 0; i < Dim; ++i)
      _num_cells[i] = static_cast<std::size_t>(num_cells[i]);
    init();
  }

  grid_coordinate_translator(const math::vector_n<Dim>& center,
                             const math::vector_n<Dim>& extent,
                             const grid_uindex& num_cells)
    :_center{center}, _extent{extent}
  {
    _num_cells = num_cells;
    init();
  }

  grid_index operator()(const math::vector_n<Dim>& pos) const
  {
    grid_index result;

    for(std::size_t i = 0; i < Dim; ++i)
      result[i] = static_cast<long long int>((pos[i] - _min_corner[i]) / _cell_size[i]);

    return result;
  }

  inline bool is_within_bounds(const grid_index& idx) const
  {
    for(std::size_t i = 0; i < Dim; ++i)
      if(idx[i] < 0 || static_cast<std::size_t>(idx[i]) >= _num_cells[i])
        return false;
    return true;
  }

  static
  std::array<std::size_t, Dim> unsigned_grid_index(const grid_index& idx)
  {
    std::array<std::size_t, Dim> result;
    for(std::size_t i = 0; i < Dim; ++i)
      result[i] = static_cast<std::size_t>(idx[i]);
    return result;
  }

  math::vector_n<Dim> get_cell_min_coordinates(const grid_index& index) const
  {
    math::vector_n<Dim> result;

    for(std::size_t i = 0; i < Dim; ++i)
      result[i] = _min_corner[i] + index[i] * _cell_size[i];

    return result;
  }

  void clamp_grid_index_to_edge(grid_index& idx) const
  {
    for(std::size_t i = 0; i < Dim; ++i)
    {
      if(idx[i] < 0)
        idx[i] = 0;
      else if(static_cast<std::size_t>(idx[i]) >= _num_cells[i])
        idx[i] = _num_cells[i] - 1;
    }
  }

  math::vector_n<Dim> get_cell_center_coordinates(const grid_index& index) const
  {
    math::vector_n<Dim> result = this->get_cell_min_coordinates(index);

    for(std::size_t i = 0; i < Dim; ++i)
      result[i] += 0.5 * _cell_size[i];

    return result;
  }

  const grid_uindex& get_num_cells() const
  {
    return _num_cells;
  }

  math::vector_n<Dim> get_cell_sizes() const
  {
    return _cell_size;
  }

  const math::vector_n<Dim>& get_grid_min_corner() const
  {
    return _min_corner;
  }

  math::vector_n<Dim> get_grid_max_corner() const
  {
    return _min_corner + _extent;
  }

private:
  void init()
  {
    for(std::size_t i = 0; i < Dim; ++i)
    {
      _cell_size[i] = _extent[i] / static_cast<math::scalar>(_num_cells[i]);
      _min_corner[i] = _center[i] - _cell_size[i] * static_cast<math::scalar>(_num_cells[i])/ 2.;
    }
  }

  math::vector_n<Dim> _center;
  math::vector_n<Dim> _extent;
  math::vector_n<Dim> _cell_size;
  grid_uindex _num_cells;
  math::vector_n<Dim> _min_corner;
};

}
}

#endif
