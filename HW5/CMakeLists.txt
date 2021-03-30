cmake_minimum_required(VERSION 3.0)

# set the project name
project(NeuralNetwork)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find Eigen
find_package(Eigen3 REQUIRED)

# add the executable
add_executable(neuralnet ../src/neuralnet.cpp)

# link library
target_include_directories(neuralnet PUBLIC ${EIGEN3_INCLUDE_DIR})
