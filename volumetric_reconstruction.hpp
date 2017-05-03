#ifndef VOLUMETRIC_RECONSTRUCTION_HPP
#define VOLUMETRIC_RECONSTRUCTION_HPP

#include <vector>

#include "qcl.hpp"
#include "math.hpp"
#include "async_io.hpp"
#include "quantity.hpp"

namespace illcrawl {

struct volume_cutout
{
  math::vector3 center;
  math::vector3 extent;

  volume_cutout(const math::vector3& volume_center,
                const math::vector3& volume_extent)
    : center(volume_center), extent(volume_extent)
  {}
};

class volumetric_reconstruction
{
public:
  using result_scalar = float;
  using nearest_neighbor_list = cl_float8;
  using result_vector4 = cl_float4;
  using result_vector3 = cl_float3;
  using result_vector2 = cl_float2;

  const std::size_t blocksize = 1000000;

  volumetric_reconstruction(const qcl::device_context_ptr& ctx,
                            const volume_cutout& render_volume,
                            const H5::DataSet& coordinates,
                            const H5::DataSet& smoothing_lengths)
    : _ctx{ctx},
      _render_volume{render_volume},
      _coordinates{coordinates},
      _smoothing_lengths{smoothing_lengths}
  {

  }

  void run(const std::vector<result_vector4>& evaluation_points,
           const reconstruction_quantity::quantity& quantity_to_reconstruct)
  {
    std::vector<H5::DataSet> streamed_quantities{
      _coordinates,
      _smoothing_lengths
    };

    for(const H5::DataSet& quantity : quantity_to_reconstruct.get_required_datasets())
      streamed_quantities.push_back(quantity);

    io::async_dataset_streamer<math::scalar> streamer{streamed_quantities};
  }

private:
  qcl::device_context_ptr _ctx;
  volume_cutout _render_volume;

  H5::DataSet _coordinates;
  H5::DataSet _smoothing_lengths;
};

}

#endif
