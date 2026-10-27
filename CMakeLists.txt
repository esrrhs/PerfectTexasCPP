cmake_minimum_required(VERSION 3.16)
project(PerfectTexasCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /utf-8 /bigobj)
else()
  add_compile_options(-Wall -Wextra)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

# LibTorch：默认找 ~/libtorch，也可以用 -DCMAKE_PREFIX_PATH=/path/to/libtorch 指定。
if(NOT DEFINED Torch_DIR AND EXISTS "$ENV{HOME}/libtorch/share/cmake/Torch")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
endif()
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

add_library(perfecttexas_core STATIC
  src/hu.cpp
  src/features.cpp
  src/nn.cpp
  src/rollout.cpp
  src/learner.cpp
  src/eval.cpp
  src/slumbot.cpp
)
target_include_directories(perfecttexas_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(perfecttexas_core PUBLIC ${TORCH_LIBRARIES} Threads::Threads CURL::libcurl)
target_compile_options(perfecttexas_core PUBLIC ${TORCH_CXX_FLAGS})

add_executable(perfecttexas src/main.cpp)
target_link_libraries(perfecttexas PRIVATE perfecttexas_core)

enable_testing()
add_executable(perfecttexas_tests tests/test_main.cpp)
target_link_libraries(perfecttexas_tests PRIVATE perfecttexas_core)
add_test(NAME perfecttexas_tests COMMAND perfecttexas_tests)
