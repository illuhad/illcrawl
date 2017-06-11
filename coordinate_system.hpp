#ifndef COORDINATE_SYSTEM_HPP
#define COORDINATE_SYSTEM_HPP

#include "grid.hpp"
#include "math.hpp"

namespace illcrawl {
namespace coordinate_system{

/// Maps a point from a periodic coordinate system to
/// the periodic frame described by a box. This is done
/// by adding or subtracting the periodic length to
/// each component until the point lies in the given box
/// (if possible).
/// \param render_volume_bounding_box_min The corner
/// of the box with minimum coordinate values
/// \param render_volume_bounding_box_max The corner
/// of the box with maximum coordinate values
/// \param perdiodic_wraparound_size The length
/// of a period for each dimension
/// \param additional_tolerance The size of an additional
/// layer around the box to increase its size.
/// \param coordinates the coordinates of the point to process.
template<std::size_t N>
void correct_periodicity(const math::vector_n<N>& render_volume_bounding_box_min,
                         const math::vector_n<N>& render_volume_bounding_box_max,
                         const math::vector_n<N>& periodic_wraparound_size,
                         math::scalar additional_tolerance,
                         math::vector_n<N>& coordinates)
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

/// Maps a point from a periodic coordinate system to
/// the periodic frame described by a box. This is done
/// by adding or subtracting the periodic length to
/// each component until the point lies in the given box
/// (if possible). The box is described by the grid
/// given by a \c grid_translator object.
/// \param grid_translator Describes the grid - the grid's min and
/// max corner coordinates will be used to define the box.
/// \param perdiodic_wraparound_size The length
/// of a period for each dimension
/// \param additional_tolerance The size of an additional
/// layer around the box to increase its size.
/// \param coordinates the coordinates of the point to process.
template<std::size_t N>
void correct_periodicity(const util::grid_coordinate_translator<N>& grid_translator,
                         const math::vector_n<N>& periodic_wraparound_size,
                         math::scalar additional_tolerance,
                         math::vector_n<N>& coordinates)
{
  correct_periodicity(grid_translator.get_grid_min_corner(),
                      grid_translator.get_grid_max_corner(),
                      periodic_wraparound_size,
                      additional_tolerance,
                      coordinates);
}

/// Maps a point from a periodic coordinate system to
/// the periodic frame in which a given pivot point is contained.
/// This is done by adding or subtracting the periodic length to
/// each component until the distance from the point to the pivot
/// point is minimal.
/// \param perdiodic_wraparound_size The length
/// of a period for each dimension
/// \param pivot The pivot point
/// \param coordinates the coordinates of the point to process.
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
