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
#include <map>
#include <unordered_map>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>


#include "hdf5_io.hpp"
#include "async_io.hpp"
#include "coordinate_system.hpp"

const std::size_t blocksize = 10000000;
const illcrawl::math::vector3 periodic_length = {{75000.,75000.,75000.}};

class dataset_descriptor
{
public:
  dataset_descriptor(const std::string& group_name,
                     const std::string& dataset_name)
    : _group_name{group_name},
      _dataset_name{dataset_name}
  {}

  dataset_descriptor(const std::string& identifier)
    : _separator{":"}
  {
    _group_name = "PartType0";
    _dataset_name = identifier;

    std::size_t separator_pos = identifier.find(_separator);
    if(separator_pos != std::string::npos)
    {
      _group_name = identifier.substr(0, separator_pos);
      _dataset_name = identifier.substr(separator_pos + 1);
    }
  }

  std::string get_identifier() const
  {
    return get_group_name() + _separator + get_dataset_name();
  }

  const std::string& get_group_name() const
  {
    return _group_name;
  }

  const std::string& get_dataset_name() const
  {
    return _dataset_name;
  }
private:
  const std::string _separator;

  std::string _group_name;
  std::string _dataset_name;
};

class dataset_collection
{
public:
  dataset_collection(const std::vector<dataset_descriptor>& datasets)
    : _descriptors(datasets)
  {
    for(const auto& descriptor : datasets)
    {
      _datasets[descriptor.get_group_name()].push_back(descriptor.get_dataset_name());
    }
    for(const auto& group_dataset_pair : _datasets)
      _groups.push_back(group_dataset_pair.first);
  }

  const std::vector<std::string>& get_groups() const
  {
    return _groups;
  }

  const std::vector<dataset_descriptor>& get_dataset_descriptors() const
  {
    return _descriptors;
  }

  const std::vector<std::string>& get_datasets(const std::string& group) const
  {
    auto it = _datasets.find(group);

    if(it == _datasets.end())
      throw std::invalid_argument{"Queried group does not exist"};

    return it->second;
  }
private:
  std::vector<dataset_descriptor> _descriptors;
  std::vector<std::string> _groups;
  std::unordered_map<std::string, std::vector<std::string>> _datasets;
};

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
    illcrawl::coordinate_system::correct_periodicity(periodic_length,
                                                     _center,
                                                     pos);

    math::vector3 R = pos - _center;
    return math::dot(R,R) < _radius * _radius;
  }

private:
  illcrawl::math::vector3 _center;
  illcrawl::math::scalar _radius;
};

class box_selector : public selector
{
public:
  box_selector(const illcrawl::math::vector3 min_corner,
               const illcrawl::math::vector3 max_corner)
    : _min_corner(min_corner), _max_corner(max_corner)
  {
    using namespace illcrawl;
    _center = 0.5 * (_min_corner + _max_corner);
  }

  virtual ~box_selector() {}


  virtual
  bool operator()(const illcrawl::math::vector3& particle_position) const override
  {
    illcrawl::math::vector3 pos = particle_position;
    illcrawl::coordinate_system::correct_periodicity(periodic_length,
                                                     _center,
                                                     pos);

    for(std::size_t i = 0; i < 3; ++i)
      if(particle_position[i] < _min_corner[i] ||
         particle_position[i] > _max_corner[i])
        return false;

    return true;
  }

private:
  illcrawl::math::vector3 _min_corner;
  illcrawl::math::vector3 _max_corner;
  illcrawl::math::vector3 _center;
};


class cube_selector : public box_selector
{
public:
  cube_selector(const illcrawl::math::vector3& center,
                illcrawl::math::scalar sidelength)
    : box_selector{get_min_corner(center,sidelength),
                   get_max_corner(center,sidelength)}
  {}

  virtual ~cube_selector(){}

private:
  static illcrawl::math::vector3 get_min_corner(const illcrawl::math::vector3& center,
                                                illcrawl::math::scalar sidelength)
  {
    illcrawl::math::vector3 result = center;

    for(std::size_t i = 0; i < 3; ++i)
      result[i] -= 0.5 * sidelength;

    return result;
  }

  static illcrawl::math::vector3 get_max_corner(const illcrawl::math::vector3& center,
                                                illcrawl::math::scalar sidelength)
  {
    illcrawl::math::vector3 result = center;

    for(std::size_t i = 0; i < 3; ++i)
      result[i] += 0.5 * sidelength;

    return result;
  }
};


class selected_particle_storage
{
public:
  selected_particle_storage(const dataset_collection& field_names)
    : _field_names(field_names)
  {
  }

  using dataset_storage = std::vector<illcrawl::math::scalar>;

  ~selected_particle_storage(){}

  void operator()(const std::string& group_name,
                  const std::string& dataset_name,
                  const std::vector<illcrawl::math::scalar>& data)
  {
    std::string dataset_key =
        dataset_descriptor{group_name, dataset_name}.get_identifier();

    if(_row_widths[dataset_key] == 0)
      _row_widths[dataset_key] = static_cast<hsize_t>(data.size());

    // Make sure that the width of this row equals the width of previous
    // rows from the same dataset
    assert(_row_widths[dataset_key] == data.size());

    std::vector<illcrawl::math::scalar>& dataset_storage =
        this->_stored_datasets[dataset_key];

    for(std::size_t i = 0; i < data.size(); ++i)
      dataset_storage.push_back(data[i]);
  }

  void save_particles(const std::string& filename) const
  {
    illcrawl::io::hdf5_writer<illcrawl::math::scalar> writer{filename};

    for(std::string group : _field_names.get_groups())
    {
      writer.add_group(group);

      for(std::string dataset : _field_names.get_datasets(group))
      {
        std::string dataset_key =
            dataset_descriptor{group, dataset}.get_identifier();

        std::vector<hsize_t> shape;
        auto dataset_iterator = _stored_datasets.find(dataset_key);
        auto row_width_iterator = _row_widths.find(dataset_key);

        if(dataset_iterator == _stored_datasets.end() ||
           row_width_iterator == _row_widths.end())
        {
          // If we have not stored any data related to this dataset, we simply
          // write an empty dataset to the output file.
          writer.add_dataset(group, dataset, std::vector<illcrawl::math::scalar>());
        }
        else
        {
          hsize_t row_width = row_width_iterator->second;
          hsize_t num_rows = dataset_iterator->second.size() / row_width;

          assert(dataset_iterator->second.size() % static_cast<std::size_t>(row_width) == 0);

          std::vector<hsize_t> shape {num_rows, row_width};

          writer.add_dataset(group, dataset, shape, dataset_iterator->second);
        }
      }
    }

  }

  std::size_t get_num_stored_particles(const std::string& group) const
  {
    assert(_field_names.get_datasets(group).size() > 0);

    std::string first_dataset = _field_names.get_datasets(group)[0];
    std::string first_dataset_key =
        dataset_descriptor{group, first_dataset}.get_identifier();

    std::size_t result = 0;

    auto storage_iterator = _stored_datasets.find(first_dataset_key);
    auto row_size_iterator = _row_widths.find(first_dataset_key);

    if(storage_iterator != _stored_datasets.end() &&
       row_size_iterator != _row_widths.end())
    {
      assert(row_size_iterator->second != 0);
      result = storage_iterator->second.size() / row_size_iterator->second;
    }

    return result;
  }


private:
  dataset_collection _field_names;
  std::unordered_map<std::string, dataset_storage> _stored_datasets;
  std::unordered_map<std::string, hsize_t> _row_widths;
};

class filter
{
public:
  using selector_list = std::vector<std::pair<selector*,selected_particle_storage*>>;

  filter(const dataset_collection& extracted_datasets,
         const selector_list& selectors)
    : _extracted_datasets(extracted_datasets),
      _selectors(selectors),
      _num_selected_particles{0}
  {
  }

  void operator()(const std::string& snapshot_prefix,
                  int num_parts)
  {
    using illcrawl::math::scalar;
    using illcrawl::math::vector3;

    bool no_parts = false;
    if(num_parts <= 0)
    {
      no_parts = true;
      num_parts = 1;
    }

    const std::vector<std::string>& groups = _extracted_datasets.get_groups();
    if(groups.size() == 0)
      throw std::invalid_argument{"No data to extract."};

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
        groups[0]
      };

      for(std::size_t group_id = 0; group_id  < groups.size(); ++group_id)
      {
        std::string group_name = groups[group_id];
        loader.select_group(groups[group_id]);



        std::vector<H5::DataSet> streamed_fields =
        {
          {loader.get_dataset(illcrawl::io::illustris_data_loader::get_coordinate_identifier())}
        };

        const std::vector<std::string>& dataset_names
            = _extracted_datasets.get_datasets(group_name);

        for(const auto& dataset_name : dataset_names)
        {
          streamed_fields.push_back(loader.get_dataset(dataset_name));
        }

        illcrawl::io::async_dataset_streamer<scalar> streamer{streamed_fields};

        std::vector<illcrawl::math::scalar> extracted_data_buffer;
        extracted_data_buffer.reserve(3);

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

            bool is_particle_selected = false;

            // Iterate over all filters, and check if the data row (=particle)
            // should be selected and stored, according to the filter
            for(auto& selector_handler_pair :  _selectors)
            {
              if((*selector_handler_pair.first)(coordinates))
              {
                is_particle_selected = true;
                // If the selector decides to include the particle based on its
                // coordinates, call the data handler for each data field.
                assert(dataset_names.size() + 1 == access.get_num_datasets());
                for(std::size_t dataset_id = 0;
                    dataset_id < dataset_names.size();
                    ++dataset_id)
                {
                  std::string dataset_name = dataset_names[dataset_id];

                  // Switch to the new dataset
                  access.select_dataset(dataset_id + 1);
                  extracted_data_buffer.clear();

                  // Extract all elements of this data row and store in
                  // extracted_data_buffer
                  for(std::size_t j = 0;
                      j < access.get_dataset_row_size(dataset_id + 1);
                      ++j)
                    extracted_data_buffer.push_back(access(current_block, i, j));

                  (*selector_handler_pair.second)(group_name,
                                                  dataset_name,
                                                  extracted_data_buffer);

                }

              }
            }
            if(is_particle_selected)
              ++_num_selected_particles;
          }
        });
      }

      std::size_t selector_id = 0;
      for(const auto& selector_storage_pair : _selectors)
      {
        std::cout << "Filter " << selector_id << " has currently retrieved:" << std::endl;
        for(const auto& group_name : _extracted_datasets.get_groups())
          std::cout << "   from group " << group_name << ": "
                    << selector_storage_pair.second->get_num_stored_particles(group_name)
                    << " data rows" << std::endl;

        ++selector_id;
      }
    }
  }

private:

  dataset_collection _extracted_datasets;
  selector_list _selectors;
  std::size_t _num_selected_particles;
};


int main(int argc, char** argv)
{

  namespace po = boost::program_options;

  po::options_description options{"Allowed options"};
  std::vector<std::string> sphere_filters;
  std::vector<std::string> cube_filters;
  std::vector<std::string> raw_extracted_fields;

  options.add_options()
      ("snapshot_prefix,p", po::value<std::string>()->default_value("snapshot_"), "set prefix for the filename of the snapshot chunks. "
                                       "E.g., if the snapshots are named snapshot_0-i.hdf5,"
                                       "with i being the chunk number, the prefix is snapshot_0-")
      ("num_snapshot_parts,n", po::value<int>(), "The number of snapshot chunks.")
      ("extract,e", po::value<std::vector<std::string>>(&raw_extracted_fields),
                                       "fields which are extracted. The format is hdf5_group:dataset, e.g. PartType0:Coordinates")
      ("sphere_filter", po::value<std::vector<std::string>>(&sphere_filters),
                                       "define a selection sphere. Format: --sphere_filter x,y,z,r,output_file")
      ("cube_filter", po::value<std::vector<std::string>>(&cube_filters),
                                       "define a selection cube. Format: --cube_filter x,y,z,half_sidelength,output_file (with the center coordinates x,y,z)")
      ("help,h", "print help message");

  try
  {

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options)
                                                 .run(),
              vm);

    if(vm.count("help"))
    {
      std::cout << options << std::endl;
      return 0;
    }

    po::notify(vm);

    std::string snapshot_prefix = vm["snapshot_prefix"].as<std::string>();
    int num_snapshot_parts = vm["num_snapshot_parts"].as<int>();

    std::cout << "Looking at " << num_snapshot_parts
              << " chunks with prefix " << snapshot_prefix << std::endl;

    std::vector<dataset_descriptor> extracted_fields;
    for(const auto& extracted_field : raw_extracted_fields)
      extracted_fields.push_back(dataset_descriptor{extracted_field});

    dataset_collection datasets{extracted_fields};

    for(const auto& group : datasets.get_groups())
    {
      std::cout << "Extracting from group: " << group << std::endl;
      for(const auto& dataset: datasets.get_datasets(group))
        std::cout << "   " << dataset << std::endl;
    }

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
                           new selected_particle_storage{datasets}
                         });

      selector_handler_pairs.push_back(std::make_pair(selectors.back().get(),
                                                      handlers.back().get()));
    }

    for(const std::string& cube_filter_description : cube_filters)
    {
      std::vector<std::string> parts;
      boost::split(parts, cube_filter_description, boost::is_any_of(",;"));
      if(parts.size() != 5)
        throw std::invalid_argument{"Invalid cube filter description: "+cube_filter_description};


      illcrawl::math::scalar x = std::stod(parts[0]);
      illcrawl::math::scalar y = std::stod(parts[1]);
      illcrawl::math::scalar z = std::stod(parts[2]);
      illcrawl::math::scalar sidelength = 2.0 * std::stod(parts[3]);

      illcrawl::math::vector3 cube_center = {{x,y,z}};
      std::string output_file = parts[4];

      std::cout << "Registering cube filter with center "
                << x << ", " << y << ", " << z
                << " and sidelength " << sidelength
                << ". Output file will be " << output_file
                << std::endl;

      output_files.push_back(output_file);


      selectors.push_back(std::unique_ptr<selector>{
                            new cube_selector{cube_center, sidelength}
                          });
      handlers.push_back(std::unique_ptr<selected_particle_storage>{
                           new selected_particle_storage{datasets}
                         });

      selector_handler_pairs.push_back(std::make_pair(selectors.back().get(),
                                                      handlers.back().get()));
    }

    filter data_filter{datasets, selector_handler_pairs};
    data_filter(snapshot_prefix, num_snapshot_parts);


    std::cout << "Saving results..." << std::endl;
    for(std::size_t i = 0; i < handlers.size(); ++i)
      handlers[i].get()->save_particles(output_files[i]);

  }
  catch(boost::bad_any_cast& e)
  {
    std::cout << options << std::endl;
  }
  catch(boost::program_options::error& e)
  {
    std::cout << "Invalid command line: " << e.what() << std::endl;
    std::cout << options << std::endl;
  }
  catch (std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
}
