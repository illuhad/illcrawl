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

#include <string>
#include <ostream>

namespace illcrawl {
namespace util {

class tree_formatter
{
public:
  tree_formatter(const std::string& tree_name)
  : _tree_name(tree_name) {}

  class node
  {
  public:
    node() = default;

    node(const node& other)
    : _sub_nodes(other._sub_nodes),
      _content(other._content.str())
    {
    }

    node& operator=(node other)
    {
      swap(other);
      return *this;
    }

    void swap(node& other)
    {
      std::swap(this->_sub_nodes, other._sub_nodes);
      this->_content.swap(other._content);
    }

    template<class T>
    void append_content(const T& data)
    {
      _content << data;
    }

    std::ostream& get_stream()
    {
      return _content;
    }

    template<class T>
    void add_node(const T& name, const node& subnode)
    {
      std::stringstream sstr;
      sstr << name;
      _sub_nodes.push_back(std::make_pair(sstr.str(), subnode));
    }

    std::string get_content() const
    {
      return _content.str();
    }

    typedef std::vector<std::pair<std::string,node>>::iterator subnode_iterator;
    typedef std::vector<std::pair<std::string,node>>::const_iterator const_subnode_iterator;

    subnode_iterator begin_subnodes()
    { return _sub_nodes.begin(); }

    const_subnode_iterator begin_subnodes() const
    { return _sub_nodes.begin(); }

    subnode_iterator end_subnodes()
    { return _sub_nodes.end(); }

    const_subnode_iterator end_subnodes() const
    { return _sub_nodes.end(); }

    void print(const std::string& tree_name, std::ostream& ostr) const
    {
      ostr << tree_name << std::endl;
      print_impl(ostr, " ", true);
    }
  private:
    void print_impl(std::ostream& ostr, const std::string& current_prefix, bool is_last_node) const
    {
      std::string current_line;

      std::string content_prefix = "   ";
      if(!_sub_nodes.empty())
        content_prefix = "|  ";

      while(std::getline(_content, current_line))
      {
        ostr << current_prefix << content_prefix << current_line << std::endl;
      }

      std::string sublevel_prefix = current_prefix + "| ";
      for(std::size_t i = 0; i < _sub_nodes.size(); ++i)
      {
        if(i != _sub_nodes.size() - 1)
        {
          ostr << current_prefix << "+   " << std::endl
               << current_prefix << "|`+ " << _sub_nodes[i].first << std::endl;

          _sub_nodes[i].second.print_impl(ostr, sublevel_prefix, false);
        }
        else
        {
          ostr << current_prefix << "+   " << std::endl
               << current_prefix << " `+ " << _sub_nodes[i].first << std::endl;
          _sub_nodes[i].second.print_impl(ostr, current_prefix + "  ", true);
        }
      }
    }

    std::vector<std::pair<std::string,node>> _sub_nodes;
    mutable std::stringstream _content;
  };

  node& get_root()
  { return _root; }

  const node& get_root() const
  { return _root; }

  const std::string& get_tree_name() const
  {
    return _tree_name;
  }
private:
  node _root;
  std::string _tree_name;
};

std::ostream& operator<<(std::ostream& ostr, const tree_formatter& tree)
{
  tree.get_root().print(tree.get_tree_name(), ostr);

  return ostr;
}

}
}

#endif
