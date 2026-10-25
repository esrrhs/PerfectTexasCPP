cmake_minimum_required(VERSION 3.16)
project(texas_cfr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /utf-8)
else()
  add_compile_options(-Wall -Wextra)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

add_library(texas_cfr INTERFACE)
target_include_directories(texas_cfr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 可选：接入 texas_algorithm 的查表牌力后端（翻牌后牌力百分位，约 280ns/次）。
# 不开启时用内置的 evaluate7 启发式，精度较低但零依赖。
option(TEXAS_CFR_WITH_ALGORITHM "Use texas_algorithm lookup tables for postflop strength" OFF)
set(TEXAS_ALGORITHM_DIR "" CACHE PATH "Path to texas_algorithm/cpp directory")

if(TEXAS_CFR_WITH_ALGORITHM)
  if(NOT TEXAS_ALGORITHM_DIR)
    message(FATAL_ERROR "TEXAS_CFR_WITH_ALGORITHM=ON 需要同时指定 -DTEXAS_ALGORITHM_DIR=<texas_algorithm/cpp>")
  endif()
  add_subdirectory(${TEXAS_ALGORITHM_DIR} ${CMAKE_BINARY_DIR}/texas_algorithm)
  target_compile_definitions(texas_cfr INTERFACE TEXAS_CFR_WITH_ALGORITHM=1)
  target_link_libraries(texas_cfr INTERFACE texas_algorithm)
  message(STATUS "texas_cfr: 启用 texas_algorithm 查表牌力后端")
endif()

add_executable(texas_cli src/main.cpp)
target_link_libraries(texas_cli PRIVATE texas_cfr)

enable_testing()
add_executable(texas_tests tests/test_main.cpp)
target_link_libraries(texas_tests PRIVATE texas_cfr)
add_test(NAME texas_tests COMMAND texas_tests)
