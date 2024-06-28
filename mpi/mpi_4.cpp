#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <cstdlib>
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

void multiplyMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int arraySize, int rank, int size) {
    int localSize = arraySize / size;
    int startRow = rank * localSize;
    int endRow = startRow + localSize;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < arraySize; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < arraySize; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> arraySizes = {10, 100, 1000};

    for (int arraySize : arraySizes) {
        std::vector<std::vector<double>> A(arraySize, std::vector<double>(arraySize));
        std::vector<std::vector<double>> B(arraySize, std::vector<double>(arraySize));
        std::vector<std::vector<double>> C(arraySize, std::vector<double>(arraySize));

        if (rank == 0) {
            A = generateRandomMatrix(arraySize);
            B = generateRandomMatrix(arraySize);
        }

        MPI_Bcast(&A[0][0], arraySize * arraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[0][0], arraySize * arraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<std::vector<double>> localC(arraySize/size, std::vector<double>(arraySize, 0.0));
        auto start = std::chrono::high_resolution_clock::now();

        multiplyMatrices(A, B, localC, arraySize, rank, size);

        MPI_Gather(&localC[0][0], (arraySize * arraySize) / size, MPI_DOUBLE, &C[0][0], (arraySize * arraySize) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedTime = end - start;

            std::cout << "Array size: " << arraySize  << ", Time: " << std::fixed << std::setprecision(10) << elapsedTime.count() << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}
