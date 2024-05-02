#include <iostream>
#include <vector>

#include "hlslib/xilinx/OpenCL.h"
#include "hlslib/xilinx/Utility.h"

int main(int argc, char **argv) {

    bool emulation = false;
    bool verify = true;
    hlslib::UnsetEnvironmentVariable("XCL_EMULATION_MODE");
    hlslib::SetEnvironmentVariable("XCL_EMULATION_MODE", "hw_emu");
    std::string path = "tracking_hw_emu.xclbin";

    std::vector<float> observation = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> quad_input (4, 0);

    try {
        std::cout << "Initializing OpenCL context...\n" << std::flush;
        hlslib::ocl::Context context;

        std::cout << "Programming device...\n" << std::flush;
        auto program = context.MakeProgram(path);

        std::cout << "Initializing device memory...\n" << std::flush;
        auto observation_device = context.MakeBuffer<float, hlslib::ocl::Access::read>(hlslib::ocl::MemoryBank::bank1, observation.cbegin(), observation.cend());
        auto input_device = context.MakeBuffer<float, hlslib::ocl::Access::write>(hlslib::ocl::MemoryBank::bank1, quad_input.cbegin(), quad_input.cbegin());

        observation_device.CopyFromHost(observation.cbegin());
        input_device.CopyFromHost(quad_input.cbegin());

        std::cout << "Creating kernel...\n" << std::flush;
        auto kernel = program.MakeKernel("tracking", observation_device, input_device);

        std::cout << "Executing kernel...\n" << std::flush;
        const auto elapsed = kernel.ExecuteTask();

        std::cout << "Copying back result...\n" << std::flush;
        input_device.CopyToHost(quad_input.begin());

    } catch (std::runtime_error const &err) {
        std::cerr << "Execution failed with error: \"" << err.what() << "\"."
                << std::endl;
        return 1;
    }

    std::cout << "Getting result...\n" << std::flush;
    printf("Inputs %f %f %f %f", quad_input[0], quad_input[1], quad_input[2], quad_input[3]);

    std::cout << "Done." << std::endl;

  return 0;
}