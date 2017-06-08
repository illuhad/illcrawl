
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-ignored-attributes -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ffast-math -march=native -Wall -Wno-ignored-attributes -Wextra -Wno-unused-parameter")
#set(CMAKE_CXX_COMPILER mpic++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpic++)

cmake_minimum_required (VERSION 3.1)
project (illcrawl)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release"
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(GLUT REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCL REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
find_package(Boost
  1.61      # Minimum or EXACT version e.g. 1.36.0
  REQUIRED             # Faiil with error if Boost is not found
  mpi serialization program_options)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS})

include_directories(${PROJECT_BINARY_DIR} ${OpenCL_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${MPI_C_INCLUDE_PATH})

add_executable(illcrawl illcrawl.cpp chandra.cpp)
add_executable(illcrawl_extract_fits_slice illcrawl_extract_fits_slice.cpp)
add_executable(illcrawl_filter_snapshot illcrawl_filter_snapshot.cpp)

target_link_libraries (illcrawl ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenCL_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}  ${Boost_MPI_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} cfitsio)
target_link_libraries (illcrawl_extract_fits_slice ${OpenCL_LIBRARIES} cfitsio)
target_link_libraries (illcrawl_filter_snapshot  ${CMAKE_THREAD_LIBS_INIT} ${HDF5_CXX_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY})
# Copy CL sources
add_custom_command(TARGET illcrawl POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/*.cl $<TARGET_FILE_DIR:illcrawl>)

#add_custom_command(TARGET illcrawl POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                       ${CMAKE_SOURCE_DIR}/*.cl_hpp $<TARGET_FILE_DIR:illcrawl>)



