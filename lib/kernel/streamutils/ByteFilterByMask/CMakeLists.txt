cmake_minimum_required(VERSION 3.10)
project(ByteFilterByMask)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx512f")

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core irreader)

add_executable(ByteFilterByMask main.cpp idisa_builder.cpp)
target_link_libraries(ByteFilterByMask ${llvm_libs})

