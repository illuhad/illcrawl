#ifndef INTERPOLATION_TREE
#define INTERPOLATION_TREE

#include <vector>
#include <array>

#include "math.hpp"
#include "qcl.hpp"

namespace illcrawl {

class sparse_interpolation_tree
{
public:
  using particle = cl_float4;
  using children_list = cl_int8;
  using particle_counter = cl_uint;
  using scalar = cl_float;
  using coordinate = cl_float4;

  using global_cell_id = cl_int;
  using subcell_id = cl_int;

  sparse_interpolation_tree(const std::vector<particle>& particles,
                            const qcl::device_context_ptr& ctx)
    : _ctx{ctx},
      _kernel{ctx->get_kernel("tree_interpolation")},
      _is_first_run{true}
  {
    assert(ctx != nullptr);

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
        if(particles[i].s[j] < min_coordinates[j])
          min_coordinates[j] = particles[i].s[j];

        if(particles[i].s[j] > max_coordinates[j])
          max_coordinates[j] = particles[i].s[j];
      }
    }


    _root_diameter = std::max(max_coordinates[0]-min_coordinates[0],
                     std::max(max_coordinates[1]-min_coordinates[1],
                              max_coordinates[2]-min_coordinates[2])) + 0.1f;
    assert(_root_diameter > 0.0f);

    for(std::size_t i = 0; i < 3; ++i)
      _center.s[i] = 0.5 * (min_coordinates[i] + max_coordinates[i]);

    // Create root
    this->add_cell();

    for(const particle& p : particles)
    {
      // Insert particle into root node
      insert_particle(0, {{_center.s[0],_center.s[1],_center.s[2]}}, _root_diameter, p);
    }

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
    std::cout << "Tree ready\n";

  }

  void evaluate_tree(const cl::Buffer& evaluation_points,
                     const cl::Buffer& value_sum_state,
                     const cl::Buffer& weight_sum_state,
                     const cl::Buffer& out,
                     std::size_t num_points,
                     scalar opening_angle,
                     cl::Event* evt) const
  {
    qcl::kernel_argument_list arguments{_kernel};
    arguments.push(static_cast<cl_int>(_is_first_run));
    arguments.push(_particles_buffer);
    arguments.push(_subcells_buffer);
    arguments.push(_num_particles_buffer);
    arguments.push(_mean_coordinates_buffer);
    arguments.push(_center);
    arguments.push(_root_diameter);
    arguments.push(opening_angle);
    arguments.push(evaluation_points);
    arguments.push(static_cast<cl_int>(num_points));
    arguments.push(value_sum_state);
    arguments.push(weight_sum_state);
    arguments.push(out);


    cl_int err = _ctx->get_command_queue().enqueueNDRangeKernel(*_kernel,
                                                   cl::NullRange,
                                                   cl::NDRange(math::make_multiple_of(
                                                                 local_size,
                                                                 num_points)),
                                                   cl::NDRange(local_size),
                                                   &_tree_ready_events,
                                                   evt);
    qcl::check_cl_error(err, "Could not enqueue tree interpolation kernel.");
    _is_first_run = false;
  }

  void evaluate_single_point(const math::vector3& position,
                               scalar opening_angle,
                               scalar& value_sum,
                               scalar& weight_sum) const
  {
    math::vector3 root_center = {{_center.s[0],_center.s[1],_center.s[2]}};
    evaluate_single_point(position,
                          0, root_center, _root_diameter,
                          opening_angle,
                          value_sum, weight_sum);
  }

private:

  inline bool can_node_be_approximated(const math::vector3& evaluation_point,
                                       const math::vector3& cell_coordinate,
                                       scalar cell_diameter,
                                       scalar opening_angle) const
  {
    math::vector3 R = evaluation_point - cell_coordinate;

    return cell_diameter * cell_diameter / math::dot(R,R) < opening_angle * opening_angle;
  }

  void evaluate_single_point(const math::vector3& position,
                             global_cell_id cell,
                             const math::vector3& cell_coordinate,
                             math::scalar cell_diameter,
                             scalar opening_angle,
                             scalar& value_sum,
                             scalar& weight_sum) const
  {
    particle p = _particle_for_cell[cell];
    for(int i = 0; i < 3; ++i)
      assert(p.s[i] >= (cell_coordinate[i] - 0.5*cell_diameter) &&
             p.s[i] <= (cell_coordinate[i] + 0.5*cell_diameter));

    if(is_cell_leaf(cell)
     || can_node_be_approximated(position, cell_coordinate, cell_diameter, opening_angle))
    {
      math::vector3 particle_pos = {{p.s[0], p.s[1], p.s[2]}};
      math::vector3 R = position - particle_pos;
      math::vector3 mean_pos = {{_mean_coordinates[cell].s[0],
                                 _mean_coordinates[cell].s[1],
                                 _mean_coordinates[cell].s[2]}};
      math::vector3 R_weights = position - mean_pos;

      math::scalar r2_inv = 1.0 / math::dot(R,R);
      math::scalar weight_inv = 1.0 / math::dot(R_weights, R_weights);

      value_sum += p.s[3] * r2_inv * r2_inv;
      weight_sum += _num_particles[cell] * weight_inv * weight_inv;
    }
    else
    {
      children_list children = _subcells[cell];
      for(std::size_t i = 0; i < 8; ++i)
      {
        if(cell_exists(children.s[i]))
        {
          math::vector3 subcell_center = get_subcell_center(cell_coordinate,
                                                         cell_diameter,
                                                         i);

          evaluate_single_point(position,
                                children.s[i],
                                subcell_center,
                                0.5 * cell_diameter,
                                opening_angle,
                                value_sum, weight_sum);
        }

      }
    }
  }

  static constexpr std::size_t local_size = 256;
  const children_list empty_children_list = {{-1, -1, -1, -1,
                                              -1, -1, -1, -1}};

  void finalize_cells()
  {
    _mean_coordinates.resize(_particle_for_cell.size());
    finalize_cells(0);
  }

  void finalize_cells(global_cell_id cell)
  {
    assert(cell_exists(cell));

    children_list subcells = _subcells[cell];

    if(is_cell_leaf(cell))
    {
      // For 1 particle, the monopole equals the particle itself, so no further
      // work is necessary
      _mean_coordinates[cell] = get_particle_coordinate(_particle_for_cell[cell]);
      _num_particles[cell] = 1;
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
            assert(cell < _mean_coordinates.size() && child_id < _mean_coordinates.size());
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
                       const math::vector3& cell_coordinate,
                       math::scalar cell_diameter,
                       particle p)
  {
    for(int i = 0; i < 3; ++i)
    {
      assert(p.s[i] >= (cell_coordinate[i] - 0.5*cell_diameter - 0.1) &&
             p.s[i] <= (cell_coordinate[i] + 0.5*cell_diameter + 0.1));
    }

    assert(cell_exists(cell));

    if(_num_particles[cell] == 0)
    {
      // This should only happen for the root node
      assert(cell == 0);
      _particle_for_cell[cell] = p;
      _num_particles[cell] = 1;
    }
    else
    {

      // Move already present particle to subcells, if the cell has been a leaf until
      // now - otherwise it will already have been moved
      if(is_cell_leaf(cell))
      {
        particle old_particle = _particle_for_cell[cell];

        subcell_id target_subcell = get_subcell_id(old_particle, cell_coordinate);
        assert(target_subcell >= 0 && target_subcell < 8);

        // Create new leaf with the old particle
        global_cell_id new_leaf = add_cell(old_particle);

        assert(!cell_exists(_subcells[cell].s[target_subcell]));
        _subcells[cell].s[target_subcell] = new_leaf;
      }

      subcell_id local_subcell_id = get_subcell_id(p, cell_coordinate);
      assert(local_subcell_id >= 0 && local_subcell_id < 8);

      global_cell_id global_subcell_id =
          static_cast<global_cell_id>(_subcells[cell].s[local_subcell_id]);

      if(cell_exists(global_subcell_id))
      {
        // Insert particle into subcell
        math::vector3 subcell_center = get_subcell_center(cell_coordinate,
                                                       cell_diameter,
                                                       local_subcell_id);

        insert_particle(global_subcell_id, subcell_center, 0.5 * cell_diameter, p);
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
    coordinate result = p;
    result.s[3] = 0.0f;
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
  math::vector3 get_subcell_center(const math::vector3& parent_center,
                                   const scalar parent_diameter,
                                   subcell_id subcell) const
  {
    static std::array<std::array<int, 3>, 8> directional_sign =
    {{
       {{-1, -1, -1}},
       {{-1, -1,  1}},
       {{-1,  1, -1}},
       {{-1,  1,  1}},
       {{ 1, -1, -1}},
       {{ 1, -1,  1}},
       {{ 1,  1, -1}},
       {{ 1,  1,  1}}
    }};

    scalar step = 0.25 * parent_diameter;

    math::vector3 result = parent_center;
    for(std::size_t i = 0; i < 3; ++i)
      result[i] += directional_sign[subcell][i] * step;

    return result;
  }

  inline
  subcell_id get_subcell_id(const particle& particle_coord,
                             const math::vector3& cell_coord) const
  {
    subcell_id result = 0;

    if(particle_coord.s[0] > cell_coord[0])
      result += 4;
    if(particle_coord.s[1] > cell_coord[1])
      result += 2;
    if(particle_coord.s[2] > cell_coord[2])
      result += 1;

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

  qcl::kernel_ptr _kernel;

  mutable bool _is_first_run;

};

}

#endif
