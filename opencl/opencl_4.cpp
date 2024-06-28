#include <iostream>
#include <vector>
#include <chrono>
#include <CL/opencl.hpp>
#include <cstdlib>
#include <iomanip>
#include <fstream>


std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

int main(int argc, char* argv[]) {
    std::vector<int> arraySizes = {10, 100, 1000};
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

    std::ifstream file("kernel_4.cl");
    std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources({sourceCode});
    cl::Program program(context, sources);
    program.build(devices);

    for (auto arraySize : arraySizes) {
        std::vector<std::vector<double>> A = generateRandomMatrix(arraySize);
        std::vector<std::vector<double>> B = generateRandomMatrix(arraySize);
        std::vector<std::vector<double>> C(arraySize, std::vector<double>(arraySize, 0.0));

        std::vector<double> flatA(arraySize * arraySize);
        std::vector<double> flatB(arraySize * arraySize);
        std::vector<double> flatC(arraySize * arraySize, 0.0);

        for (int i = 0; i < arraySize; ++i) {
            for (int j = 0; j < arraySize; ++j) {
                flatA[i * arraySize + j] = A[i][j];
                flatB[i * arraySize + j] = B[i][j];
            }
        }

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flatA.size(), flatA.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flatB.size(), flatB.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(double) * flatC.size());

        cl::Kernel kernel(program, "matrixMultiply");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, arraySize);

        cl::NDRange global(arraySize, arraySize);
        cl::NDRange local(numThreads, numThreads);

        auto start = std::chrono::high_resolution_clock::now();
        
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = end - start;

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(double) * flatC.size(), flatC.data());


        for (int i = 0; i < arraySize; ++i) {
            for (int j = 0; j < arraySize; ++j) {
                C[i][j] = flatC[i * arraySize + j];
            }
        }

        std::cout << "Array size: " << arraySize << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
    }
}
