#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <cmath>
#include <cstdint>
#include <iomanip>

double f(int32_t i, int32_t j) {
    return cos(i) * i + sin(j) * j;
}

double derivative(double x1, double x2) {
    return (x1 - x2) / 2.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<size_t> arraySizes = {10, 1000, 10000};

    for (auto arraySize : arraySizes) {
        size_t localSize = arraySize / size;
        size_t startRow = rank * localSize;
        size_t endRow = (rank + 1) * localSize;

        std::vector<std::vector<double>> A(localSize, std::vector<double>(arraySize));
        std::vector<std::vector<double>> B(localSize, std::vector<double>(arraySize, 0.0));

        for (size_t i = 0; i < localSize; ++i) {
            for (size_t j = 0; j < arraySize; ++j) {
                A[i][j] = f(static_cast<int32_t>(startRow + i), static_cast<int32_t>(j));
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 1; i < localSize - 1; ++i) {
            for (size_t j = 0; j < arraySize; ++j) {
                B[i][j] = derivative(A[i+1][j], A[i-1][j]);
            }
        }   

        if (rank == 0) {
            for (size_t j = 0; j < arraySize; ++j) {
                B[0][j] = derivative(A[1][j], A[0][j]);
            }
        }

        if (rank == size - 1) {
            for (size_t j = 0; j < arraySize; ++j) {
                B[localSize - 1][j] = derivative(A[localSize - 1][j], A[localSize - 2][j]);
            }
        }

        if (rank == 0) {
            std::vector<std::vector<double>> resultB(arraySize, std::vector<double>(arraySize, 0.0));

            for (size_t i = 0; i < localSize; ++i) {
                for (size_t j = 0; j < arraySize; ++j) {
                    resultB[startRow + i][j] = B[i][j];
                }
            }

            for (size_t process = 1; process < size; ++process) {
                size_t procStartRow = process * localSize;
                MPI_Recv(&resultB[procStartRow][0], localSize * arraySize, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedTime = end - start;

            std::cout << "Array size: " << arraySize  << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
        } else {
            MPI_Send(&B[0][0], localSize * arraySize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
