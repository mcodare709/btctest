cmake_minimum_required(VERSION 3.20)
project(btc_perp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(btc_strategy SHARED cpp/src/strategy.cpp)
target_include_directories(btc_strategy PUBLIC cpp/include)

add_executable(btc_strategy_test cpp/tests/strategy_test.cpp)
target_link_libraries(btc_strategy_test PRIVATE btc_strategy)
