#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <mpi.h>
#include <random>
#include <iomanip>

std::vector<int32_t> random_array(int32_t min_val, int32_t max_val, size_t size) {
    std::random_device rd;
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    std::vector<int32_t> arr(size);

    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(rd);
    }
    return arr;
}

int64_t sum_vector_elements(const std::vector<int32_t>& vec) {
    int64_t sum = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum;
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<size_t> array_sizes = {10, 1000, 10000000};
    std::vector<size_t> processes = {2, 8, 16, 32, 100, 1000};

    for (auto array_size : array_sizes) {
        std::vector<int32_t> array;
        array = random_array(-10, 10, array_size);

        MPI_Barrier(MPI_COMM_WORLD);
        auto start_time = std::chrono::steady_clock::now();

        MPI_Bcast(array.data(), array_size, MPI_INT, 0, MPI_COMM_WORLD);
        int64_t partial_sum = std::accumulate(array.begin(), array.end(), 0);
        int64_t total_sum = 0;

        MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        auto end_time = std::chrono::steady_clock::now();
        if (rank == 0) {
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            std::cout << "Array size: " << array_size  << ", Time: " << std::fixed << std::setprecision(10) << elapsed_time.count() << " seconds\n";
        }
    }

    MPI_Finalize();
    return 0;
}