#include <iostream>
#include <vector>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

// g++ opencl_1.cpp -o path/to/output -lOpenCL && path/to/output <num_of_threads>

int main(int argc, char** argv) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cl::CommandQueue queue(context, devices[0]);

    std::ifstream file("kernel_1.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

    std::vector<std::string> sourceStrings{{sourceCode}};

    cl::Program::Sources sources(sourceStrings);
    cl::Program program(context, sources);

    program.build(devices);

    cl::Kernel kernel(program, "hello_world");

    int num_of_threads = 10;
    if (argc > 1){
        num_of_threads = std::atoi(argv[1]);
    }

    cl::NDRange globalSize(num_of_threads);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
    queue.finish();
}
