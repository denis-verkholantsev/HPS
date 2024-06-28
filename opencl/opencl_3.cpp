#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <CL/opencl.hpp>
#include <fstream>
#include <iomanip>

#define CL_HPP_TARGET_OPENCL_VERSION 300

double f(int32_t i, int32_t j) {
    return cos(i) * i + sin(j) * j;
}


int main(int argc, char* argv[]) {
    std::vector<size_t> arraySizes = {10, 1000, 10000};
    size_t numThreads = 8;

    if (argc > 1){
        numThreads = std::atoi(argv[1]);
    }    

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);

    std::ifstream file("kernel_3.cl");
    std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources({sourceCode});
    cl::Program program(context, sources);
    program.build(devices);

    for (auto arraySize : arraySizes) {

        std::vector<double> A(arraySize * arraySize);
        std::vector<double> B(arraySize * arraySize);

        for (size_t i = 0; i < arraySize; ++i) {
            for (size_t j = 0; j < arraySize; ++j) {
                A[i * arraySize + j] = f(static_cast<int32_t>(i), static_cast<int32_t>(j));
            }
        }

        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(double) * arraySize * arraySize);
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(double) * arraySize * arraySize);

        auto start = std::chrono::high_resolution_clock::now();

        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(double) * arraySize * arraySize, A.data());

        cl::Kernel kernel(program, "derivative");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, arraySize);

        cl::NDRange globalSize(arraySize);
        cl::NDRange localSize(numThreads);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    
        queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * arraySize * arraySize, B.data());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = end - start;

        std::cout << "Array size: " << arraySize << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
    }

    return 0;
}
