__kernel void sum(__global int* input, __global long* result) {
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    __local long local_sum;

    if (local_id == 0) {
        local_sum = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&local_sum, input[global_id]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        atomic_add(result, local_sum);
    }
}
