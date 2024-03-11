#include<iostream>
#include<mpi.h>
#include<string>

// mpic++ mpi_1.cpp -o path/to/output && mpiexec -n <number_of_processes> path/to/output

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank; // process number
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << std::string("Process " + std::to_string(rank) + ": Hello world\n");

    MPI_Finalize();
}