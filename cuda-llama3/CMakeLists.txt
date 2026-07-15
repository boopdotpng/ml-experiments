cmake_minimum_required(VERSION 3.24)
project(cuda_llama3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # This project intentionally targets the user's Blackwell RTX 5090. Explicit
  # SM 120 also avoids older CMake "native" probes falling back to SM 75.
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_executable(cuda_llama
  src/main.cpp
  src/cuda_llama.cu
)

target_compile_options(cuda_llama PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;--ptxas-options=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
