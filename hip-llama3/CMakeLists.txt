cmake_minimum_required(VERSION 3.21)
project(hip_llama3 LANGUAGES CXX HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

add_executable(hip_llama
  src/main.cpp
  src/hip_llama.hip
)

target_include_directories(hip_llama PRIVATE src)
target_compile_options(hip_llama PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O2>
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
)
