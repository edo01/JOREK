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

// Kernel now works on flattened 2D matrix
__global__
void test_kernel(int* arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index
    int total = n * n;
    if (idx < total)
    {
        int row = idx / n;
        int col = idx % n;
        arr[idx] = row * n + col;  // example: fill with linear index
    }

    // ATTENTION: in C we are filling matrix row-wise (the memory actually contains contiguous elements).
    // Since Fortran is column-major, the continuous elements for Fortran are column-wise.
    // Actually, C works on the transpose matrix of Fortran's one!!
}

extern "C"
void launch_test_kernel(int* arr, int n)
{
    int* d_arr = nullptr;
    int total = n * n;

    printf("BEFORE KERNEL LAUNCH\n");
    for (int i = 0; i < total; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    HIP_CHECK(hipMalloc(&d_arr, total * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_arr, arr,
                        total * sizeof(int),
                        hipMemcpyHostToDevice));

    int blockSize = 64;
    int gridSize  = (total + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(test_kernel,
                       dim3(gridSize),
                       dim3(blockSize),
                       0, 0,
                       d_arr, n);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(arr, d_arr,
                        total * sizeof(int),
                        hipMemcpyDeviceToHost));

    printf("AFTER KERNEL LAUNCH\n");
    for (int i = 0; i < total; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    HIP_CHECK(hipFree(d_arr));
}