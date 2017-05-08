#ifndef INTERPOLATION_TREE
#define INTERPOLATION_TREE

#include <vector>
#include <array>

#include "qcl.hpp"

namespace illcrawl {

class sparse_interpolation_tree
{
public:
  using particle = cl_float4;
  using children_list = cl_int8;
  using particle_counter = cl_uint;
  using scalar = cl_float;
  using coordinate = cl_float3;

  using global_cell_id = cl_int;
  using subcell_id = cl_int;

  sparse_interpolation_tree(const std::vector<particle>& particles,
                            const qcl::device_context_ptr& ctx)
    : _ctx{ctx}
  {
    _center = {{0.0f, 0.0f, 0.0f}};

    std::array<scalar,3> min_coordinates = {{-1.f, -1.f, -1.f}};
    std::array<scalar,3> max_coordinates = {{ 1.f,  1.f,  1.f}};

    if(particles.size() > 0)
    {
      for(std::size_t j = 0; j < 3; ++j)
      {
        min_coordinates[j] = particles[0].s[j];
        max_coordinates[j] = particles[0].s[j];
      }
    }

    for(std::size_t i = 0; i < particles.size(); ++i)
    {
      for(std::size_t j = 0; j < 3; ++j)
      {
        _center.s[j] += particles[i].s[j];
        if(particles[i].s[j] < min_coordinates[j])
          min_coordinates[j] = particles[i].s[j];

        if(particles[i].s[j] > max_coordinates[j])
          max_coordinates[j] = particles[i].s[j];
      }
    }

    for(std::size_t j = 0; j < 3; ++j)
      _center.s[j] /= static_cast<scalar>(particles.size());

    _root_diameter = std::max(max_coordinates[0]-min_coordinates[0],
                     std::max(max_coordinates[1]-min_coordinates[1],
                              max_coordinates[2]-min_coordinates[2])) + 0.1f;


    assert(_root_diameter > 0.0f);

    // Create root
    this->add_cell();

    for(const particle& p : particles)
      // Insert particle into root node
      insert_particle(0, _center, _root_diameter, p);

    // Finalize the cells recursively
    finalize_cells();

    _ctx->create_input_buffer<particle>(this->_particles_buffer,
                                        _particle_for_cell.size());
    _ctx->create_input_buffer<children_list>(this->_subcells_buffer,
                                             _subcells.size());
    _ctx->create_input_buffer<particle_counter>(this->_num_particles_buffer,
                                                _num_particles.size());
    _ctx->create_input_buffer<coordinate>(this->_mean_coordinates_buffer,
                                          _mean_coordinates.size());

    assert(_particle_for_cell.size() == _subcells.size());
    assert(_particle_for_cell.size() == _num_particles.size());
    assert(_particle_for_cell.size() == _mean_coordinates.size());


    _tree_ready_events = std::vector<cl::Event>(4);
    _ctx->memcpy_h2d_async(_particles_buffer,
                           _particle_for_cell.data(),
                           _particle_for_cell.size(),
                           &(_tree_ready_events[0]));

    _ctx->memcpy_h2d_async(_subcells_buffer,
                           _subcells.data(),
                           _subcells.size(),
                           &(_tree_ready_events[1]));

    _ctx->memcpy_h2d_async(_num_particles_buffer,
                           _num_particles.data(),
                           _num_particles.size(),
                           &(_tree_ready_events[2]));

    _ctx->memcpy_h2d_async(_mean_coordinates_buffer,
                           _mean_coordinates.data(),
                           _mean_coordinates.size(),
                           &(_tree_ready_events[3]));

  }

  void evaluate_tree(const cl::Buffer& evaluation_points,
                     const cl::Buffer& out,
                     std::size_t num_points) const
  {

  }

private:

  static constexpr children_list empty_children_list = {{-1, -1, -1, -1,
                                                         -1, -1, -1, -1}};

  void finalize_cells()
  {
    _mean_coordinates.resize(_particle_for_cell.size());
    finalize_cells(0);
  }

  void finalize_cells(global_cell_id cell)
  {
    children_list subcells = _subcells[cell];

    if(is_cell_leaf(cell))
    {
      _mean_coordinates[cell] = get_particle_coordinate(_particle_for_cell[cell]);
      _num_particles[cell] = 1;
      // For 1 particle, the monopole equals the particle itself, so no further
      // work is necessary
    }
    else
    {
      _num_particles[cell] = 0;
      _mean_coordinates[cell] = coordinate{{0.0f, 0.0f, 0.0f}};
      _particle_for_cell[cell] = particle{{0.0f, 0.0f, 0.0f, 0.0f}};
      scalar total_mass = 0.0f;

      for(std::size_t i = 0; i < 8; ++i)
      {
        // Construct multipoles and mean coordinates from subnodes
        global_cell_id child_id = subcells.s[i];
        if(cell_exists(child_id))
        {
          // Recursively finalize cells
          finalize_cells(child_id);

          particle_counter num_child_particles =  _num_particles[child_id];
          _num_particles[cell] += num_child_particles;
          particle child_particle = _particle_for_cell[child_id];
          scalar particle_mass = child_particle.s[3];

          total_mass += particle_mass;
          for(std::size_t j = 0; j < 3; ++j)
          {
            _mean_coordinates[cell].s[j] += _mean_coordinates[child_id].s[j]
                                          * num_child_particles;
            // Center of mass for multipole
            _particle_for_cell[cell].s[j] += child_particle.s[j] * particle_mass;
          }
        }
      }

      for(std::size_t i = 0; i < 3; ++i)
      {
        _mean_coordinates[cell].s[i] /= static_cast<scalar>(_num_particles[cell]);
        _particle_for_cell[cell].s[i] /= total_mass;
      }
      _particle_for_cell[cell].s[3] = total_mass;
    }
  }

  void insert_particle(global_cell_id cell,
                       const coordinate& cell_coordinate,
                       scalar cell_diameter,
                       const particle& p)
  {
    if(_num_particles[cell] == 0)
      _particle_for_cell[cell] = p;
    else
    {
      subcell_id local_subcell_id = get_subcell_id(p, cell_coordinate);

      global_cell_id global_subcell_id =
          static_cast<global_cell_id>(_subcells[cell].s[local_subcell_id]);

      // Move already present particle to subcells, if the cell has been a leaf until
      // now - otherwise it will already have been moved
      if(is_cell_leaf(cell))
      {
        particle old_particle = _particle_for_cell[cell];
        coordinate particle_coordinate = get_particle_coordinate(old_particle);
        subcell_id target_subcell = get_subcell_id(old_particle, cell_coordinate);

        // Create new leaf with the old particle
        global_cell_id new_leaf = add_cell(old_particle);

        assert(!cell_exists(_subcells[cell].s[target_subcell]));
        _subcells[cell].s[target_subcell] = new_leaf;
      }

      if(cell_exists(global_subcell_id))
      {
        // Insert particle into subcell
        coordinate subcell_center = get_subcell_center(cell_coordinate,
                                                       cell_diameter,
                                                       local_subcell_id);

        insert_particle(global_subcell_id, subcell_center, 0.5f * cell_diameter, p);
      }
      else
      {
        // Create new leaf with particle
        global_cell_id new_leaf = add_cell(p);
        _subcells[cell].s[local_subcell_id] = new_leaf;
      }


    }
  }

  inline coordinate get_particle_coordinate(const particle& p) const
  {
    coordinate result;
    for(std::size_t i = 0; i < 3; ++i)
      result.s[i] = p.s[i];
    return result;
  }

  inline bool cell_exists(global_cell_id cell) const
  {
    return cell >= 0 && cell < _particle_for_cell.size();
  }

  inline
  global_cell_id add_cell()
  {
    _particle_for_cell.push_back(particle{});
    _subcells.push_back(empty_children_list);
    _num_particles.push_back(0);

    return _particle_for_cell.size() - 1;
  }

  inline
  global_cell_id add_cell(const particle& p)
  {
    _particle_for_cell.push_back(p);
    _subcells.push_back(empty_children_list);
    _num_particles.push_back(1);

    return _particle_for_cell.size() - 1;
  }

  inline bool is_cell_leaf(global_cell_id cell) const
  {
    children_list children = _subcells[cell];
    for(std::size_t i = 0; i < 8; ++i)
      if(cell_exists(children.s[i]))
        return false;
    return true;
  }


  inline
  coordinate get_subcell_center(const coordinate& parent_center,
                                const scalar parent_diameter,
                                subcell_id subcell) const
  {
    static std::array<std::array<int, 3>, 8> directional_sign =
    {{
       {{-1, -1, -1}}, // subcell_id = 0
       {{ 1, -1, -1}}, // subcell_id = 1
       {{-1,  1, -1}}, // subcell_id = 2
       {{ 1,  1, -1}}, // subcell_id = 3
       {{-1, -1,  1}}, // subcell_id = 4
       {{ 1, -1,  1}}, // subcell_id = 5
       {{-1,  1,  1}}, // subcell_id = 6
       {{ 1,  1,  1}}  // subcell_id = 7
    }};

    scalar delta = 0.25f * parent_diameter;

    coordinate result = parent_center;
    for(std::size_t i = 0; i < 3; ++i)
      result.s[i] += directional_sign[subcell][i] * delta;

    return result;
  }

  inline
  subcell_id get_subcell_id(const particle& particle_coord,
                             const coordinate& cell_coord) const
  {
    subcell_id result = 0;

    if(particle_coord.s[0] > cell_coord.s[0])
      result += 1;
    if(particle_coord.s[1] > cell_coord.s[1])
      result += 2;
    if(particle_coord.s[2] > cell_coord.s[2])
      result += 4;

    return result;
  }

  qcl::device_context_ptr _ctx;

  coordinate _center;
  scalar _root_diameter;
  std::vector<particle> _particle_for_cell;

  std::vector<children_list> _subcells;

  std::vector<particle_counter> _num_particles;
  std::vector<coordinate> _mean_coordinates;

  cl::Buffer _particles_buffer;
  cl::Buffer _subcells_buffer;
  cl::Buffer _num_particles_buffer;
  cl::Buffer _mean_coordinates_buffer;

  std::vector<cl::Event> _tree_ready_events;

};

}

#endif
