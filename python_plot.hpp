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

#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <sstream>

#define PYTHON2 "python2"


#include "multi_array.hpp"

namespace illcrawl {
namespace python_plot {

class basic_figure
{
public:

  basic_figure(const std::string& identifier)
    : _id{identifier}, _script_name{identifier + "_plot.py"},
      _num_data_arrays{0}
  {
    _header = "import numpy as np\n"
              "import matplotlib.pyplot as plt\n";
  }

  void generate_script() const
  {
    std::ofstream file{_script_name.c_str(), std::ios::trunc};

    if (!file.is_open())
      throw std::runtime_error("Could not generate script file");

    file << _header;
    file << _data;
    file << _commands;
  }

  void show()
  {
    add_command("plt.show()");
  }

  void save()
  {
    add_command("plt.savefig(\"" + _id + ".png\")");
  }

  void set_title(const std::string& title)
  {
    add_command("plt.title(r'" + title + "')");
  }

  void custom_command(const std::string& python_command)
  {
    add_command(python_command);
  }

  void generate() const
  {
    generate_script();
    run_script();
  }
protected:

  void append_to_header(const std::string& line)
  {
    _header += line;
    _header += "\n";
  }

  void add_command(const std::string& line)
  {
    _commands += line;
    _commands += "\n";
  }


  template<class Input_iterator>
  std::string add_data_array(Input_iterator begin, Input_iterator end)
  {
    std::string array_name = "data";
    array_name += std::to_string(_num_data_arrays);

    _data += array_name + " = [";

    std::size_t num_elements = 0;
    for (Input_iterator it = begin;
      it != end; ++it)
    {
      _data += std::to_string(*it);
      _data += ",";

      ++num_elements;
    }
    // Replace the last "," with a "]"
    if (num_elements != 0)
      _data[_data.length() - 1] = ']';

    _data += "\n";

    ++_num_data_arrays;

    return array_name;
  }

  /// \return The id of the created data array

  template<class Container_type>
  std::string add_data_array(const Container_type& x)
  {
    return add_data_array(x.begin(), x.end());
  }

  
  template<class Matrix_type>
  std::string add_data_matrix(const Matrix_type& m)
  {
    std::string array_name = "data";
    array_name += std::to_string(_num_data_arrays);
    
    _data += array_name + " = [";

    for(std::size_t j = 0; j < m[0].size(); ++j)
    {
      _data += "[";
      for(std::size_t i = 0; i < m.size(); ++i)
      {
        _data += std::to_string(m[i][j]);
        if(i != m[j].size() - 1)
          _data += ", ";
      }
      _data += "]";
      if(j != m.size() - 1)
        _data += ", ";
    }
    _data += "]\n";
    
    ++_num_data_arrays;
    
    return array_name;
  }
  

  template<class T>
  std::string add_data_matrix(const util::multi_array<T>& m)
  {
    assert(m.get_dimension() == 2);
    std::string array_name = "data";
    array_name += std::to_string(_num_data_arrays);
    
    _data += array_name + " = [";

    for(std::size_t j = 0; j < m.get_extent_of_dimension(1); ++j)
    {
      _data += "[";
      for(std::size_t i = 0; i < m.get_extent_of_dimension(0); ++i)
      {
        std::size_t idx [] = {i,j};
        _data += std::to_string(m[idx]);
        if(i != m.get_extent_of_dimension(0) - 1)
          _data += ", ";
      }
      _data += "]";
      if(j != m.get_extent_of_dimension(1))
        _data += ", ";
    }
    _data += "]\n";
    
    ++_num_data_arrays;
    
    return array_name;
  }

private:

  void run_script() const
  {
    // TODO: Change this to a less ugly solution
    std::string command_string = PYTHON2 +
      std::string(" ") + _script_name;
    std::system(command_string.c_str());
  }

  std::string _id;
  std::string _script_name;

  std::string _header;
  std::string _data;
  std::string _commands;

  std::size_t _num_data_arrays;
};

class figure2d : public basic_figure
{
public:

  figure2d(const std::string& identifier)
    : basic_figure{identifier}
  {
  }

  template<class Input_iterator1,
           class Input_iterator2>
  void plot(Input_iterator1 x_begin, Input_iterator1 x_end,
            Input_iterator2 y_begin, Input_iterator2 y_end,
            const std::string& label = "",
            const std::string& additional_args = "")
  {
    basic_plot2d_call("plot",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, additional_args);
  }

  template<class Container1_type,
           class Container2_type>
  void plot(const Container1_type& x, const Container2_type& y,
               const std::string& label = "",
               const std::string& additional_args = "")
  {
    plot(x.begin(), x.end(),
         y.begin(), y.end(),
         label, additional_args);
  }

  template<class Input_iterator1,
           class Input_iterator2>
  void scatter(Input_iterator1 x_begin, Input_iterator1 x_end,
               Input_iterator2 y_begin, Input_iterator2 y_end,
               const std::string& label = "",
               const std::string& additional_args = "")
  {
    basic_plot2d_call("scatter",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, additional_args);
  }

  template<class Container1_type,
           class Container2_type>
  void scatter(const Container1_type& x, const Container2_type& y,
               const std::string& label = "",
               const std::string& additional_args = "")
  {
    scatter(x.begin(), x.end(),
            y.begin(), y.end(),
            label, additional_args);
  }

  template<class Input_iterator1,
           class Input_iterator2>
  void errorbar(Input_iterator1 x_begin, Input_iterator1 x_end,
                Input_iterator2 y_begin, Input_iterator2 y_end,
                const std::string& label = "",
                const std::string& additional_args = "")
  {
    basic_plot2d_call("errorbar",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, additional_args);
  }

  template<class Container1_type,
           class Container2_type>
  void errorbar(const Container1_type& x, const Container2_type& y,
                const std::string& label = "",
                const std::string& additional_args = "")
  {
    errorbar(x.begin(), x.end(), y.begin(), y.end(), label, additional_args);
  }

  template<class Input_iterator1,
           class Input_iterator2,
           class Input_iterator3>
  void errorbar_x(Input_iterator1 x_begin, Input_iterator1 x_end,
                  Input_iterator2 y_begin, Input_iterator2 y_end,
                  Input_iterator3 dx_begin, Input_iterator3 dx_end,
                  const std::string& label = "",
                  const std::string& additional_args = "")
  {
    std::string dx_data = this->add_data_array(dx_begin, dx_end);
    std::string args = additional_args;
    if (!args.empty())
      args += ", ";
    args += "xerr=";
    args += dx_data;

    basic_plot2d_call("errorbar",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, args);
  }

  template<class Container1_type,
           class Container2_type,
           class Container3_type>
  void errorbar_x(const Container1_type& x, const Container2_type& y,
                  const Container3_type& dx,
                  const std::string& label = "",
                  const std::string& additional_args = "")
  {
    errorbar_x(x.begin(), x.end(),
               y.begin(), y.end(),
               dx.begin(), dx.end(),
               label, additional_args);
  }

  template<class Input_iterator1,
           class Input_iterator2,
           class Input_iterator3>
  void errorbar_y(Input_iterator1 x_begin, Input_iterator1 x_end,
                  Input_iterator2 y_begin, Input_iterator2 y_end,
                  Input_iterator3 dy_begin, Input_iterator3 dy_end,
                  const std::string& label = "",
                  const std::string& additional_args = "")
  {
    std::string dy_data = this->add_data_array(dy_begin, dy_end);
    std::string args = additional_args;
    if (!args.empty())
      args += ", ";
    args += "yerr=";
    args += dy_data;

    basic_plot2d_call("errorbar",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, args);
  }

  template<class Container1_type,
           class Container2_type,
           class Container3_type>
  void errorbar_y(const Container1_type& x,
                  const Container2_type& y,
                  const Container3_type& dy,
                  const std::string& label = "",
                  const std::string& additional_args = "")
  {
    errorbar_y(x.begin(), x.end(),
               y.begin(), y.end(),
               dy.begin(), dy.end(),
               label, additional_args);
  }

  template<class Input_iterator1,
           class Input_iterator2,
           class Input_iterator3,
           class Input_iterator4>
  void errorbar_x_y(Input_iterator1 x_begin, Input_iterator1 x_end,
                    Input_iterator2 y_begin, Input_iterator2 y_end,
                    Input_iterator3 dx_begin, Input_iterator3 dx_end,
                    Input_iterator4 dy_begin, Input_iterator4 dy_end,
                    const std::string& label = "",
                    const std::string& additional_args = "")
  {
    std::string dx_data = this->add_data_array(dx_begin, dx_end);
    std::string dy_data = this->add_data_array(dy_begin, dy_end);
    std::string args = additional_args;
    if (!args.empty())
      args += ", ";
    args += "xerr=";
    args += dx_data;
    args += ", yerr=";
    args += dy_data;

    basic_plot2d_call("errorbar",
                      x_begin, x_end,
                      y_begin, y_end,
                      label, args);
  }

  template<class Container1_type,
           class Container2_type,
           class Container3_type,
           class Container4_type>
  void errorbar_x_y(const Container1_type& x, const Container2_type& y,
                    const Container3_type& dx,
                    const Container4_type& dy,
                    const std::string& label = "",
                    const std::string& additional_args = "")
  {
    errorbar_x_y(x.begin(), x.end(),
                 y.begin(), y.end(),
                 dx.begin(), dx.end(),
                 dy.begin(), dy.end(),
                 label, additional_args);
  }

  void legend(int location)
  {
    std::string command = "plt.legend(loc=";
    command += std::to_string(location);
    command += ")";
    this->add_command(command);
  }

  void set_x_label(const std::string& label)
  {
    std::string command = "plt.xlabel(";
    command += "r'";
    command += label;
    command += "')";
    this->add_command(command);
  }

  void set_y_label(const std::string& label)
  {
    std::string command = "plt.ylabel(";
    command += "r'";
    command += label;
    command += "')";
    this->add_command(command);
  }

  void set_semilog_x()
  {
    this->add_command("plt.semilogx()");
  }

  void set_semilog_y()
  {
    this->add_command("plt.semilogy()");
  }

  void set_loglog()
  {
    this->add_command("plt.loglog()");
  }

  template<class Coordinate_type>
  void set_x_range(Coordinate_type min, Coordinate_type max)
  {
    std::string command = "plt.xlim([";
    command += std::to_string(min);
    command += ", ";
    command += std::to_string(max);
    command += "])";

    this->add_command(command);
  }

  template<class Coordinate_type>
  void set_y_range(Coordinate_type min, Coordinate_type max)
  {
    std::string command = "plt.ylim([";
    command += std::to_string(min);
    command += ", ";
    command += std::to_string(max);
    command += "])";

    this->add_command(command);
  }
  
  template<class Container2d_type,
           class Scalar1_type, 
           class Scalar2_type,
           class Scalar3_type,
           class Scalar4_type>
  void imshow(const Container2d_type& array,
              Scalar1_type left,
              Scalar2_type bottom,
              Scalar3_type right,
              Scalar4_type top,
              const std::string& label = "",
              const std::string& additional_args = "")
  {
    std::string data = this->add_data_matrix(array);
    
    std::string extent_str = "extent=[";
    extent_str += std::to_string(left);
    extent_str += ", ";
    extent_str += std::to_string(right);
    extent_str += ", ";
    extent_str += std::to_string(bottom);
    extent_str += ", ";
    extent_str += std::to_string(top);
    extent_str += "]";
    
    std::string command = "plt.imshow(";
    command += data;
    command += ", origin='lower', ";
    command += extent_str;

    if (!label.empty())
    {
      command += ", label=r'";
      command += label;
      command += "'";
    }
    if (!additional_args.empty())
    {
      command += ", ";
      command += additional_args;
    }
    command += ")";

    this->add_command(command);
  }
  
  void show_colormap()
  {
    this->add_command("plt.colorbar()");
  }

protected:

  template<class Input_iterator1, class Input_iterator2>
  void basic_plot2d_call(const std::string& plot_command_name,
                         Input_iterator1 x_begin, Input_iterator1 x_end,
                         Input_iterator2 y_begin, Input_iterator2 y_end,
                         const std::string& label = "",
                         const std::string& additional_args = "")
  {
    std::string x_data_name = this->add_data_array(x_begin, x_end);
    std::string y_data_name = this->add_data_array(y_begin, y_end);

    std::string command = "plt.";
    command += plot_command_name;
    command += "(";
    command += x_data_name;
    command += ", ";
    command += y_data_name;

    if (!label.empty())
    {
      command += ", label=r'";
      command += label;
      command += "'";
    }
    if (!additional_args.empty())
    {
      command += ", ";
      command += additional_args;
    }
    command += ")";

    this->add_command(command);
  }
};

}
}

