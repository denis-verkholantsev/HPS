#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <omp.h>

double f(int32_t i, int32_t j) {
    return cos(i) * i + sin(j) * j;
}

double derivative(double x1, double x2) {
    return (x1 - x2) / 2.0;
}

int main(int argc, char* argv[]){
    size_t numThreads = 4;

    if (argc > 1) {
        numThreads = std::atoi(argv[1]);
    }

    omp_set_num_threads(numThreads);

    std::vector<size_t> arraySizes = {10, 1000, 10000};

    for (auto arraySize : arraySizes) {
        std::vector<std::vector<double>> A(arraySize, std::vector<double>(arraySize));
        std::vector<std::vector<double>> B(arraySize, std::vector<double>(arraySize, 0.0));

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < arraySize; ++i) {
                for (size_t j = 0; j < arraySize; ++j) {
                    A[i][j] = f(static_cast<int32_t>(i), static_cast<int32_t>(j));
                }
            }
        
        #pragma omp parallel
        {   
            for (size_t j = 0; j < arraySize; ++j) {
                    B[0][j] = derivative(A[1][j], A[0][j]);
                    B[arraySize - 1][j] = derivative(A[arraySize - 1][j], A[arraySize - 2][j]);
            }

            for (size_t i = 1; i < arraySize - 1; ++i) {
                for (size_t j = 0; j < arraySize; ++j) {
                    B[i][j] = derivative(A[i+1][j], A[i-1][j]);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = end - start;

        std::cout << "Array size: " << arraySize  << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
    }

    return 0;
}
