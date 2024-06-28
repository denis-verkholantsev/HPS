#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 300

std::vector<int32_t> randomArray(int32_t min_val, int32_t max_val, size_t size) {
    std::random_device rd;
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    std::vector<int32_t> arr(size); 

    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(rd);
    }
    return arr;
}

int64_t sumElements(const std::vector<int32_t>& vec, size_t threads) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    cl::CommandQueue queue(context, devices[0]);

    std::ifstream file("kernel_2.cl");
    std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources({sourceCode});
    cl::Program program(context, sources);

    program.build(devices);

    cl::Kernel kernel(program, "sum");
    cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vec.size() * sizeof(int32_t), (void*)vec.data());
    cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int64_t));
    
    int64_t zero = 0;
    queue.enqueueWriteBuffer(resultBuffer, CL_TRUE, 0, sizeof(int64_t), &zero);

    kernel.setArg(0, buffer);
    kernel.setArg(1, resultBuffer);

    cl::NDRange globalSize(vec.size());
    cl::NDRange localSize(threads);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();

    int64_t result = 0;
    queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(int64_t), &result);

    return result;
}

int main(int argc, char* argv[]) {
    std::vector<size_t> array_sizes = {10, 1000, 10000000};

    for (auto array_size : array_sizes) {
        auto array = randomArray(-10, 10, array_size);
        size_t numThreads = 8;

        if (argc > 1){
            numThreads = std::atoi(argv[1]);
        }
        std::cout << "array_size: " << array_size << std::endl;
        auto start = std::chrono::steady_clock::now();
        sumElements(array, numThreads);
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "threads: " << numThreads << ", time: " << std::fixed << std::setprecision(10) << elapsed.count() << " seconds\n";
        std::cout << std::endl;
    }
    return 0;
}
