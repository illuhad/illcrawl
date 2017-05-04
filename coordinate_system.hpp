#ifndef COORDINATE_SYSTEM_HPP
#define COORDINATE_SYSTEM_HPP

#include "grid.hpp"
#include "math.hpp"

namespace illcrawl {
namespace coordinate_system{

template<std::size_t N>
void correct_periodicity(const util::grid_coordinate_translator<N>& grid_translator,
                         const math::vector3& periodic_wraparound_size,
                         math::scalar smoothing_length,
                         math::vector3& coordinates)
{
  auto grid_min_coordinates = grid_translator.get_grid_min_corner();
  auto grid_max_coordinates = grid_translator.get_grid_max_corner();
  for (std::size_t j = 0; j < N; ++j)
  {
    math::scalar grid_min = grid_min_coordinates[j] - smoothing_length;
    math::scalar grid_max = grid_max_coordinates[j] + smoothing_length;

    if (math::geometry::is_within_range(coordinates[j] - periodic_wraparound_size[j],
                                      grid_min,
                                      grid_max))
      coordinates[j] -= periodic_wraparound_size[j];
    else if (math::geometry::is_within_range(coordinates[j] + periodic_wraparound_size[j],
                                           grid_min,
                                           grid_max))
      coordinates[j] += periodic_wraparound_size[j];
  }
}

}
}

#endif
