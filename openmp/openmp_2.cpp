#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <random>
#include <iomanip>

int64_t sum_vector_elements(const std::vector<int>& vec, size_t threads) {
    int64_t sum = 0;
    #pragma omp parallel shared(vec) reduction (+: sum) num_threads(threads)
    {
        # pragma omp for
        for (size_t i = 0; i < vec.size(); ++i) {
            sum += vec[i];
        }
    }
    return sum;
}

 std::vector<int32_t> random_array(int32_t minVal, int32_t maxVal, size_t size){
    std::random_device rd;
    std::uniform_int_distribution<int32_t> dist(minVal, maxVal);
    std::vector<int32_t> arr(size); 

    for(size_t i = 0; i < size; ++i){
        arr[i] = dist(rd);
    }
    return arr;
}

int main(int argc, char* argv[]) {
    std::vector<size_t> arraySizes = {10, 1000, 10000000};

    for (auto arraySize : arraySizes) {
        auto array = random_array(-100, 100, arraySize);
        size_t numThreads = 4;
        if (argc > 1) {
            numThreads = std::atoi(argv[1]);
        }

        std::cout<<"array_size: "<< arraySize << std::endl;
        auto start = std::chrono::steady_clock::now();
        sum_vector_elements(array, numThreads);
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "threads: " << numThreads << ", time: " <<std::fixed << std::setprecision(10) << elapsed.count() << " seconds\n";
    }
    return 0;
}