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

#ifndef DATA_TABLE_LOADER_HPP
#define DATA_TABLE_LOADER_HPP

#include <string>
#include <fstream>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>

#include "multi_array.hpp"

namespace illcrawl {
namespace util {

template<class T>
class data_table_loader
{
public:
  data_table_loader(const std::string& filename, const std::string delimiters="\t ")
  {
    std::ifstream file{filename.c_str()};
    if(!file.is_open())
      throw std::runtime_error("Could not load data table from file: "+filename);

    std::string line;
    std::size_t row_size = 0;
    std::vector<T> data_elements;

    while(std::getline(file, line))
    {
      // Ignore comments
      std::vector<std::string> split_line;
      if(line.find("#") != 0 && line.size() > 0)
      {
        boost::trim(line);
        boost::algorithm::split(split_line,
                                line,
                                boost::is_any_of(delimiters),
                                boost::algorithm::token_compress_on);
        if(row_size != 0)
        {
          if(split_line.size() != row_size)
            throw std::runtime_error("Invalid row in file "+filename+
                                     ": Expected "+std::to_string(row_size)+
                                     " entries in row, got "+std::to_string(split_line.size())+
                                     " (line was: \""+line+"\")");
        }
        else
          row_size = split_line.size();

        for(const std::string& str : split_line)
        {
          try
          {
            data_elements.push_back(boost::lexical_cast<T>(str));
          }
          catch(...)
          {
            throw std::runtime_error("Could not convert entry in data table "
                                     +filename
                                     +" to specified data type: "
                                     +str);
          }
        }
      }
    }

    if(row_size != 0)
    {
      assert(data_elements.size() % row_size == 0);
      assert(data_elements.size() > 0);
      _table = multi_array<T>{row_size, data_elements.size() / row_size};
      assert(_table.get_num_elements() == data_elements.size());

      std::copy(data_elements.begin(), data_elements.end(), _table.begin());
    }
  }

  const multi_array<T>& operator()() const
  {
    return _table;
  }

private:
  multi_array<T> _table;
};

}
}

#endif

