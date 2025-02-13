cmake_minimum_required(VERSION 3.4)

# Set the C++ standard to C++11 or higher
# Find the Pybind11 package
MESSAGE( "******* PYTrilinos_C++-Test Cmake script ********")

#find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)` not needed

project(example_hello_world)
add_executable(example_hello_world pybind11_test.cpp)
target_link_libraries(example_hello_world PRIVATE pybind11::embed) # here we link against pybind11::embed header if we exclude it the linking does not work anymore

project(example_scipy)
add_executable(example_scipy pybind11_test_scipy.cpp)
target_link_libraries(example_scipy PRIVATE pybind11::embed)



