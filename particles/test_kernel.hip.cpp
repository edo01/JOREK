#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(call) \
{ \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d -> %s\n", \
               __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__
void test_kernel(int* arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = i * 2;   // simple test operation
}

extern "C"
void launch_test_kernel(int* arr, int n)
{
    int *d_arr = nullptr;

    printf("BEFORE KERNEL LAUNCH\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    HIP_CHECK(hipMalloc(&d_arr, n * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_arr, arr,
                        n * sizeof(int),
                        hipMemcpyHostToDevice));

    int blockSize = 64;
    int gridSize  = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(test_kernel,
                       dim3(gridSize),
                       dim3(blockSize),
                       0, 0,
                       d_arr, n);

    HIP_CHECK(hipGetLastError());        // check launch
    HIP_CHECK(hipDeviceSynchronize());   // check execution

    HIP_CHECK(hipMemcpy(arr, d_arr,
                        n * sizeof(int),
                        hipMemcpyDeviceToHost));

    printf("AFTER KERNEL LAUNCH\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    HIP_CHECK(hipFree(d_arr));
}