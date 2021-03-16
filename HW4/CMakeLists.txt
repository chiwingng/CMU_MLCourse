cmake_minimum_required(VERSION 3.0)

# set the project name
project(LogisticRegression)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_executable(Feature ../src/feature.cpp)
add_executable(Lr ../src/lr.cpp)
