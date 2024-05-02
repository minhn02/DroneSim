#include <iostream>
#include <cstring>

#include "xrt/xrt_bo.h"
#include "experimental/xrt_xclbin.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

inline void SetEnvironmentVariable(std::string const &key,
                                   std::string const &val) {
  const auto ret = setenv(key.c_str(), val.c_str(), 1);
  if (ret != 0) {
    throw std::runtime_error("Failed to set environment variable " + key);
  }
}

inline void UnsetEnvironmentVariable(std::string const &key) {
  unsetenv(key.c_str());
}

int main(int argc, char **argv) {
    UnsetEnvironmentVariable("XCL_EMULATION_MODE");
    SetEnvironmentVariable("XCL_EMULATION_MODE", "hw_emu");

    // Read settings
    std::string binaryFile = "tracking_hw_emu.xclbin";
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "tracking", xrt::kernel::cu_access_mode::exclusive);

    std::cout << "Allocate Buffer in Global Memory\n";
    auto boObs = xrt::bo(device, 12*4, krnl.group_id(0)); //Match kernel arguments to RTL kernel
    auto boResult = xrt::bo(device, 4*4, krnl.group_id(1));

    // Map the contents of the buffer object into host memory
    auto boObs_map = boObs.map<float*>();
    auto boResult_map = boResult.map<float*>();
    std::fill(boObs_map, boObs_map + 12*4, 0);
    std::fill(boResult_map, boResult_map + 4*4, 0);

    // Create the test data
    for (int i = 0; i < 12; ++i) {
        boObs_map[i] = i;
        boResult_map[i] = 0;
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";
    boObs.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel\n";
    auto run = krnl(boObs, boResult);
    run.wait();

    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    boResult.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Validate results
    printf("got results: %f %f %f %f\n", boResult_map[0], boResult_map[1], boResult_map[2], boResult_map[3]);

    return 0;
}