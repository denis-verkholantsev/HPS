__kernel void matrixMultiply(__global double* A, __global double* B, __global double* C, int size) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    double sum = 0.0;
    
    for (int k = 0; k < size; ++k) {
        sum += A[i * size + k] * B[k * size + j];
    }
    
    C[i * size + j] = sum;
}