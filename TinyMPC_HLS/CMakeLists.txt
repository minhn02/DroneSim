cmake_minimum_required(VERSION 3.0)
project(TinyMPC_HLS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vitis REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(include/matlib)
include_directories(include/tinympc)
add_subdirectory(src)
add_subdirectory(examples)

include_directories(${CMAKE_BINARY_DIR} SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS} )

add_executable(trackingHost src/trackingHost.cpp)
target_link_libraries(trackingHost ${Vitis_LIBRARIES})

# Will populate the "hw", "hw_emu", and "synthesis" targets
add_vitis_kernel(tracking
                 FILES examples/quadrotor_tracking_hls.cpp src/matlib/matlib.cpp src/tinympc/admm.cpp src/tinympc/variables.cpp
                 INCLUDE_DIRS include hlslib/include include/tinympc include/matlib examples)

add_vitis_program(tracking 
                  xilinx_u250_gen3x16_xdma_4_1_202210_1
                  CLOCK 400
                  DEBUGGING ON
                  PROFILING ON)