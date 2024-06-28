#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <iomanip>

std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

void multiplyMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, size_t arraySize) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < arraySize; ++i) {
        for (size_t j = 0; j < arraySize; ++j) {
            C[i][j] = 0.0;
            for (size_t k = 0; k < arraySize; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    std::vector<int> arraySizes = {10, 100, 1000};
    int numThreads = 8;

    if (argc > 1) {
        numThreads = std::atoi(argv[1]);
    }

    omp_set_num_threads(numThreads);

    for (auto arraySize : arraySizes) {
        std::vector<std::vector<double>> A = generateRandomMatrix(arraySize);
        std::vector<std::vector<double>> B = generateRandomMatrix(arraySize);
        std::vector<std::vector<double>> C(arraySize, std::vector<double>(arraySize));

        auto start = std::chrono::high_resolution_clock::now();

        multiplyMatrices(A, B, C, arraySize);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = end - start;

        std::cout << "Array size: " << arraySize << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
    }

    return 0;
}
