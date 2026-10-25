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

add_executable(texas_cli src/main.cpp)
target_link_libraries(texas_cli PRIVATE texas_cfr)

enable_testing()
add_executable(texas_tests tests/test_main.cpp)
target_link_libraries(texas_tests PRIVATE texas_cfr)
add_test(NAME texas_tests COMMAND texas_tests)
