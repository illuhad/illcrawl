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


#ifndef TREE_OSTREAM
#define TREE_OSTREAM

#include <vector>
#include <string>
#include <ostream>
#include <sstream>

namespace illcrawl {
namespace util {

class tree_formatter
{
public:
  tree_formatter(const std::string& tree_name);

  class node
  {
  public:
    node() = default;

    node(const node& other);
    node& operator=(node other);
    void swap(node& other);

    template<class T>
    void append_content(const T& data)
    {
      _content << data;
    }

    std::ostream& get_stream();

    template<class T>
    void add_node(const T& name, const node& subnode)
    {
      std::stringstream sstr;
      sstr << name;
      _sub_nodes.push_back(std::make_pair(sstr.str(), subnode));
    }

    std::string get_content() const;

    typedef std::vector<std::pair<std::string,node>>::iterator subnode_iterator;
    typedef std::vector<std::pair<std::string,node>>::const_iterator const_subnode_iterator;

    subnode_iterator begin_subnodes();
    const_subnode_iterator begin_subnodes() const;

    subnode_iterator end_subnodes();
    const_subnode_iterator end_subnodes() const;

    void print(const std::string& tree_name, std::ostream& ostr) const;
  private:
    void print_impl(std::ostream& ostr, const std::string& current_prefix, bool is_last_node) const;

    std::vector<std::pair<std::string,node>> _sub_nodes;
    mutable std::stringstream _content;
  };

  node& get_root();

  const node& get_root() const;

  const std::string& get_tree_name() const;
private:
  node _root;
  std::string _tree_name;
};

std::ostream& operator<<(std::ostream& ostr, const tree_formatter& tree);

}
}

#endif
