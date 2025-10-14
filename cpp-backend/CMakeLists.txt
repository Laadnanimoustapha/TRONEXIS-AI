cmake_minimum_required(VERSION 3.16)
project(nexus_backend LANGUAGES CXX)

# Enable C++20 for modern language features like std::jthread
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the executable from a single translation unit
add_executable(nexus_backend
    src/main.cpp
)

# On Windows / MSVC, ensure UTF-8 source processing for consistency
if (MSVC)
    target_compile_options(nexus_backend PRIVATE /utf-8)
endif()