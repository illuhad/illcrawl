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

#include "tree_ostream.hpp"

namespace illcrawl {
namespace util {



tree_formatter::tree_formatter(const std::string& tree_name)
  : _tree_name(tree_name)
{}

tree_formatter::node&
tree_formatter::get_root()
{
  return _root;
}

const tree_formatter::node&
tree_formatter::get_root() const
{ return _root; }

const std::string&
tree_formatter::get_tree_name() const
{
  return _tree_name;
}


tree_formatter::node::node(const node& other)
    : _sub_nodes(other._sub_nodes),
      _content(other._content.str())
  {
  }

tree_formatter::node&
tree_formatter::node::operator=(node other)
{
  swap(other);
  return *this;
}

void
tree_formatter::node::swap(node& other)
{
  std::swap(this->_sub_nodes, other._sub_nodes);

  // Swapping stringstreams seems to be buggy
  // with older versions of gcc, hence we do it manually
  std::string my_content = this->_content.str();
  this->_content.clear();
  this->_content.str(other._content.str());
  other._content.clear();
  other._content.str(my_content);
}


std::ostream&
tree_formatter::node::get_stream()
{
  return _content;
}


std::string
tree_formatter::node::get_content() const
{
  return _content.str();
}

tree_formatter::node::subnode_iterator
tree_formatter::node::begin_subnodes()
{ return _sub_nodes.begin(); }

tree_formatter::node::const_subnode_iterator
tree_formatter::node::begin_subnodes() const
{ return _sub_nodes.begin(); }

tree_formatter::node::subnode_iterator
tree_formatter::node::end_subnodes()
{ return _sub_nodes.end(); }

tree_formatter::node::const_subnode_iterator
tree_formatter::node::end_subnodes() const
{ return _sub_nodes.end(); }

void
tree_formatter::node::print(const std::string& tree_name,
                            std::ostream& ostr) const
{
  ostr << tree_name << std::endl;
  print_impl(ostr, " ", true);
}

void
tree_formatter::node::print_impl(std::ostream& ostr,
                                 const std::string& current_prefix,
                                 bool is_last_node) const
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

std::ostream& operator<<(std::ostream& ostr, const tree_formatter& tree)
{
  tree.get_root().print(tree.get_tree_name(), ostr);

  return ostr;
}


}
}
