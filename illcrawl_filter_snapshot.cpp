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

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>


#include "hdf5_io.hpp"
#include "async_io.hpp"
#include "coordinate_system.hpp"

const std::size_t blocksize = 500000;

class selector
{
public:
  virtual ~selector(){}
  virtual bool operator()(const illcrawl::math::vector3& particle_position) const = 0;
};

class sphere_selector : public selector
{
public:
  sphere_selector(const illcrawl::math::vector3& center,
                   illcrawl::math::scalar radius)
    : _center(center), _radius{radius}
  {}

  virtual ~sphere_selector(){}

  virtual
  bool operator()(const illcrawl::math::vector3& particle_position) const override
  {
    using namespace illcrawl;

    math::vector3 pos = particle_position;
    illcrawl::coordinate_system::correct_periodicity({{75000.,75000.,75000.}},
                                                     _center,
                                                     pos);

    math::vector3 R = pos - _center;
    return math::dot(R,R) < _radius * _radius;
  }

private:
  illcrawl::math::vector3 _center;
  illcrawl::math::scalar _radius;
};



class selected_particle_storage
{
public:
  selected_particle_storage(const std::vector<std::string>& field_names)
    : _field_names{field_names},
      _stored_datasets(field_names.size() + 1),
      _dataset_shapes(field_names.size() + 1)
  {}

  ~selected_particle_storage(){}

  void operator()(illcrawl::io::buffer_accessor<illcrawl::math::scalar>& access,
                  const illcrawl::io::async_dataset_streamer<illcrawl::math::scalar>::const_iterator&
                    current_block,
                  std::size_t selected_row)
  {
    assert(_stored_datasets.size() == access.get_num_datasets());

    for(std::size_t i = 0; i < _stored_datasets.size(); ++i)
    {
      // Set the dataset shape, if it has not yet been set. This will
      // only be the case for the first row that has been read.
      // In exactly this case, the dimension of dataset_shapes[i] will still be
      // 0 (because it is still uninitialized), and therefore different
      // than the read dataset shape (which can never have a dimension of 0).
      // In this case, set the dataset shape.
      if(_dataset_shapes[i].size() != access.get_dataset_shape(i).size())
        _dataset_shapes[i] = access.get_dataset_shape(i);

      access.select_dataset(i);
      for(std::size_t j = 0; j < access.get_dataset_row_size(i); ++j)
      {
        _stored_datasets[i].push_back(access(current_block, selected_row, j));
      }
    }
  }

  void clear()
  {
    _stored_datasets.clear();
  }

  void save_particles(const std::string& filename, unsigned part_type) const
  {
    illcrawl::io::hdf5_writer<illcrawl::math::scalar> writer{filename};

    std::string group = "PartType"+std::to_string(part_type);
    writer.add_group(group);

    for(std::size_t i = 0; i < _stored_datasets.size(); ++i)
    {
      std::string field_name = "Coordinates";

      if(i != 0)
        field_name = _field_names[i - 1];

      std::vector<hsize_t> shape = _dataset_shapes[i];
      if(shape.size() == 0)
        shape.push_back(0);

      writer.add_dataset(group, field_name, shape, _stored_datasets[i]);
    }

  }

  std::size_t get_num_stored_particles() const
  {
    std::size_t num_particles = 0;
    if(_stored_datasets.size() > 0)
      num_particles = _stored_datasets.front().size();
    return num_particles;
  }
private:
  std::vector<std::string> _field_names;
  std::vector<std::vector<illcrawl::math::scalar>> _stored_datasets;
  std::vector<std::vector<hsize_t>> _dataset_shapes;
};

class filter
{
public:
  using selector_list = std::vector<std::pair<selector*,selected_particle_storage*>>;

  filter(const std::vector<std::string>& included_fields,
         const selector_list& selectors)
    : _included_fields(included_fields),
      _selectors(selectors),
      _num_selected_particles{0}
  {
  }

  void operator()(const std::string& snapshot_prefix,
                  int num_parts,
                  unsigned part_type)
  {
    using illcrawl::math::scalar;
    using illcrawl::math::vector3;

    bool no_parts = false;
    if(num_parts <= 0)
    {
      no_parts = true;
      num_parts = 1;
    }

    for(int i = 0; i < num_parts; ++i)
    {
      std::string part_filename = snapshot_prefix;
      if(!no_parts)
      {
        part_filename += std::to_string(i);
        part_filename += ".hdf5";
      }

      std::cout << "Processing " << part_filename << std::endl;

      illcrawl::io::illustris_data_loader loader{
        part_filename,
        part_type
      };

      std::vector<H5::DataSet> streamed_fields =
      {
        {loader.get_dataset(illcrawl::io::illustris_data_loader::get_coordinate_identifier())}
      };

      for(const auto& field : _included_fields)
        streamed_fields.push_back(loader.get_dataset(field));

      illcrawl::io::async_dataset_streamer<scalar> streamer{streamed_fields};

      illcrawl::io::buffer_accessor<scalar> access = streamer.create_buffer_accessor();
      illcrawl::io::async_for_each_block(streamer.begin_row_blocks(blocksize),
                                         streamer.end_row_blocks(),
                                         [&](const illcrawl::io::async_dataset_streamer<scalar>::const_iterator& current_block)
      {
        for(std::size_t i = 0; i < current_block.get_num_available_rows(); ++i)
        {
          // Coordinates always come first
          access.select_dataset(0);

          vector3 coordinates;
          for(std::size_t j = 0; j < 3; ++j)
            coordinates[j] = access(current_block, i, j);

          for(auto& selector_handler_pair :  _selectors)
          if((*selector_handler_pair.first)(coordinates))
          {
            (*selector_handler_pair.second)(access, current_block, i);
            ++_num_selected_particles;
          }
        }
      });

      std::cout << _num_selected_particles
                << " particles are selected." << std::endl;
    }
  }

private:
  std::vector<std::string> _included_fields;
  selector_list _selectors;
  std::size_t _num_selected_particles;
};


int main(int argc, char** argv)
{

  namespace po = boost::program_options;


  try
  {
    po::options_description options{"Allowed options"};
    std::vector<std::string> sphere_filters;
    std::vector<std::string> extracted_fields;

    options.add_options()
        ("snapshot_prefix,p", po::value<std::string>()->default_value("snapshot_"), "set prefix for the filename of the snapshot chunks. "
                                         "E.g., if the snapshots are named snapshot_0-i.hdf5,"
                                         "with i being the chunk number, the prefix is snapshot_0-")
        ("num_snapshot_parts,n", po::value<int>(), "The number of snapshot chunks.")
        ("extract,e", po::value<std::vector<std::string>>(&extracted_fields),
                                         "fields which are extracted (apart from coordinates -- they are always extracted)")
        ("sphere_filter", po::value<std::vector<std::string>>(&sphere_filters),
                                         "define a selection sphere. Format: --sphere_filter x,y,z,r,output_file")
        ("help,h", "print help message")
    ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options)
                                                 .run(),
              vm);

    po::notify(vm);

    if(vm.count("help"))
    {
      std::cout << options << std::endl;
      return 0;
    }

    std::string snapshot_prefix = vm["snapshot_prefix"].as<std::string>();
    int num_snapshot_parts = vm["num_snapshot_parts"].as<int>();

    std::cout << "Looking at " << num_snapshot_parts
              << " chunks with prefix " << snapshot_prefix << std::endl;

    for(const auto& extracted_field : extracted_fields)
      std::cout << "Extracting field: " << extracted_field << std::endl;


    std::vector<std::unique_ptr<selector>> selectors;
    std::vector<std::unique_ptr<selected_particle_storage>> handlers;
    std::vector<std::string> output_files;
    filter::selector_list selector_handler_pairs;

    for(const std::string& sphere_filter_description : sphere_filters)
    {
      std::vector<std::string> parts;
      boost::split(parts, sphere_filter_description, boost::is_any_of(",;"));
      if(parts.size() != 5)
        throw std::invalid_argument{"Invalid sphere filter description: "+sphere_filter_description};

      illcrawl::math::scalar x = std::stod(parts[0]);
      illcrawl::math::scalar y = std::stod(parts[1]);
      illcrawl::math::scalar z = std::stod(parts[2]);
      illcrawl::math::scalar r = std::stod(parts[3]);

      illcrawl::math::vector3 sphere_center = {{x,y,z}};
      std::string output_file = parts[4];

      std::cout << "Registering spherical filter at "
                << x << ", " << y << ", " << z
                << " with radius " << r
                << ". Output file will be " << output_file
                << std::endl;

      output_files.push_back(output_file);

      selectors.push_back(std::unique_ptr<selector>{
                            new sphere_selector{sphere_center, r}
                          });
      handlers.push_back(std::unique_ptr<selected_particle_storage>{
                           new selected_particle_storage{extracted_fields}
                         });

      selector_handler_pairs.push_back(std::make_pair(selectors.back().get(),
                                                      handlers.back().get()));
    }

    filter data_filter{extracted_fields, selector_handler_pairs};
    data_filter(snapshot_prefix, num_snapshot_parts, 0);

    for(std::size_t i = 0; i < handlers.size(); ++i)
      std::cout << "Filter " << i
                << " selected " << handlers[i]->get_num_stored_particles()
                << " data points." << std::endl;

    std::cout << "Saving results..." << std::endl;
    for(std::size_t i = 0; i < handlers.size(); ++i)
      handlers[i].get()->save_particles(output_files[i], 0);

  }
  catch (std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
}
