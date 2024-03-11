__kernel void hello_world()
{
    printf("Thread %i: Hello world\n", get_global_id(0));
}