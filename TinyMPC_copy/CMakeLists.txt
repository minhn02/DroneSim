cmake_minimum_required(VERSION 3.0)
project(TinyMPC_HLS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vitis REQUIRED)

include_directories(include/matlib)
include_directories(include/tinympc)
add_subdirectory(src)
add_subdirectory(examples)

include_directories(${Vitis_INCLUDE_DIRS})

# Will populate the "hw", "hw_emu", and "synthesis" targets
add_vitis_kernel(tracking
                 FILES examples/example_quadrotor_tracking_cpu.cpp src/matlib/matlib.cpp src/tinympc/admm.cpp src/tinympc/variables.cpp
                 INCLUDE_DIRS include hlslib/include include/tinympc include/matlib)

add_vitis_program(tracking xilinx_u250_gen3x16_xdma_4_1_202210_1)