#include<iostream>
#include<omp.h>
#include<string>

// compile: g++ -fopenmp openmp_1.cpp -o path/to/output
// run: path/to/output <num_threads>

int main(int argc, char** argv) {
    int num_threads = 0;
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        std::cout << std::string("Thread "+ std::to_string(thread) + ": Hello world\n");
    }

    return 0;
}