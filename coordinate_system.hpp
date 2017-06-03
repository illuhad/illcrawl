#ifndef COORDINATE_SYSTEM_HPP
#define COORDINATE_SYSTEM_HPP

#include "grid.hpp"
#include "math.hpp"

namespace illcrawl {
namespace coordinate_system{

template<std::size_t N>
void correct_periodicity(const math::vector_n<N>& render_volume_bounding_box_min,
                         const math::vector_n<N>& render_volume_bounding_box_max,
                         const math::vector3& periodic_wraparound_size,
                         math::scalar additional_tolerance,
                         math::vector3& coordinates)
{
  auto grid_min_coordinates = render_volume_bounding_box_min;
  auto grid_max_coordinates = render_volume_bounding_box_max;
  for (std::size_t j = 0; j < N; ++j)
  {
    math::scalar grid_min = grid_min_coordinates[j] - additional_tolerance;
    math::scalar grid_max = grid_max_coordinates[j] + additional_tolerance;

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

template<std::size_t N>
void correct_periodicity(const util::grid_coordinate_translator<N>& grid_translator,
                         const math::vector3& periodic_wraparound_size,
                         math::scalar additional_tolerance,
                         math::vector3& coordinates)
{
  correct_periodicity(grid_translator.get_grid_min_corner(),
                      grid_translator.get_grid_max_corner(),
                      periodic_wraparound_size,
                      additional_tolerance,
                      coordinates);
}


void correct_periodicity(const math::vector3& periodic_wraparound_size,
                         const math::vector3& pivot,
                         math::vector3& coordinates)
{
  for(std::size_t i = 0; i < 3; ++i)
  {
    math::vector3 candidate = coordinates;

    candidate[i] = coordinates[i] + periodic_wraparound_size[i];
    if(math::distance2(pivot, candidate) < math::distance2(pivot, coordinates))
      coordinates[i] += periodic_wraparound_size[i];
    else
    {
      candidate[i] = coordinates[i] - periodic_wraparound_size[i];
      if(math::distance2(pivot, candidate) < math::distance2(pivot, coordinates))
        coordinates[i] -= periodic_wraparound_size[i];
    }

  }
}

}
}

#endif
