# CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(AI_Prompt_System)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(prompt_app main.cpp)