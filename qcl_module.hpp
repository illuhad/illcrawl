/*
 * This file is part of QCL, a small OpenCL interface which makes it quick and
 * easy to use OpenCL.
 * 
 * Copyright (C) 2016  Aksel Alpay
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

#ifndef QCL_MODULE_HPP
#define QCL_MODULE_HPP

#include <string>
#include <boost/preprocessor/stringize.hpp>


/// Defines a module containing CL sources with a given name to identify it.
/// CL modules allow writing CL code without quotation marks and they do not
/// necessarily require seperate files.
/// A CL module must always be started by a call to \c cl_source_module_begin,
/// optionally followed by one or several calls to \c cl_include_module,
/// \c cl_source, \c cl_preprocessor etc.
/// It must always be terminated by a call to \c cl_source_module_end.
#define cl_source_module_begin(module_name)                  \
class module_name                                            \
{                                                            \
  static std::string _concatenate_source_parts()             \
  {                                                          \
    std::string result;                         

/// If your CL module needs to use functions from a different module,
/// cl_include_module will include this module and hence grant access to these
/// functions.
#define cl_include_module(include_name) result += include_name::source();
#define cl_single_line(src) result += BOOST_PP_STRINGIZE(src); result += "\n";
#define cl_preprocessor(src) cl_single_line(src)
#define cl_preprocessor_directive(directive, src) \
  result += directive;                            \
  result += " ";                                  \
  result += BOOST_PP_STRINGIZE(src);              \
  result += "\n";
#define cl_define(src) cl_preprocessor_directive("#define",src)
#define cl_ifdef(cond) cl_preprocessor_directive("#ifdef", cond)
#define cl_ifndef(cond) cl_preprocessor_directive("#ifndef", cond)
#define cl_if(cond) cl_preprocessor_directive("#if", cond)
#define cl_elif(cond) cl_preprocessor_directive("#elif", cond)
#define cl_else(cond) cl_preprocessor_directive("#else", src)
#define cl_endif(cond) cl_preprocessor_directive("#endif", src)
#define cl_source(src) result += BOOST_PP_STRINGIZE(src);

/// Ends the source module
#define cl_source_module_end()                                     \
    return result;                                                 \
  }                                                                \
public:                                                            \
  static std::string source()                                      \
  {                                                                \
    return _concatenate_source_parts();                            \
  }                                                                \
};



#endif
