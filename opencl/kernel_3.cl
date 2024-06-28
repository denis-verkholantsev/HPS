__kernel void compute_derivative(__global double* A, __global double* B, const int size) {
    int gid = get_global_id(0);
    if (gid > 0 && gid < size - 1) {
        for (int j = 0; j < size; ++j) {
            B[gid * size + j] = (A[(gid + 1) * size + j] - A[(gid - 1) * size + j]) / 2.0;
        }
    }
    else if (gid == 0) {
        for (int j = 0; j < size; ++j) {
            B[gid * size + j] = (A[(gid + 1) * size + j] - A[gid * size + j]) / 2.0;
        }
    }
    else if (gid == size - 1) {
        for (int j = 0; j < size; ++j) {
            B[gid * size + j] = (A[gid * size + j] - A[(gid - 1) * size + j]) / 2.0;
        }
    }
}
