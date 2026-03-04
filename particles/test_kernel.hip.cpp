#include <hip/hip_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Nested derived-type interoperability test
//
// Layout (matches Fortran bind(C) types):
//   test_inner_type  { int m; }
//   test_outer_type  { test_inner_type* inners; int* result; int n; int arr_size; }
//
// Computation:  result[i * arr_size + j] = j * inners[i].m
//
// Fortran verification (column-major, 1-based):  result(j+1, i+1) == j * m_i
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) \
{ \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d -> %s\n", \
               __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct test_inner_type {
    int m;  // scale factor
};

struct test_outer_type {
    test_inner_type* inners;  // array of n inner structs
    int*             result;  // flat output buffer [n * arr_size]
    int              n;       // number of inner structs
    int              arr_size;
};

// Each block handles one inner struct (blockIdx.x == i).
// Each thread handles one element (threadIdx.x == j).
__global__
void nested_kernel(test_inner_type* inners, int* result, int arr_size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (j < arr_size)
        result[i * arr_size + j] = j * inners[i].m;
}

extern "C"
void launch_test_kernel(struct test_outer_type outer)
{
    int n        = outer.n;
    int arr_size = outer.arr_size;

    // --- print host input ---
    printf("[C] BEFORE KERNEL: inner scale factors: ");
    for (int i = 0; i < n; ++i)
        printf("m[%d]=%d  ", i, outer.inners[i].m);
    printf("\n");

    // --- copy inners to device ---
    test_inner_type* d_inners = nullptr;
    HIP_CHECK(hipMalloc(&d_inners, n * sizeof(test_inner_type)));
    HIP_CHECK(hipMemcpy(d_inners, outer.inners, n * sizeof(test_inner_type),
                        hipMemcpyHostToDevice));

    // --- allocate device result ---
    int* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, n * arr_size * sizeof(int)));

    // --- launch: n blocks x arr_size threads ---
    hipLaunchKernelGGL(nested_kernel,
                       dim3(n), dim3(arr_size), 0, 0,
                       d_inners, d_result, arr_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // --- copy result back ---
    HIP_CHECK(hipMemcpy(outer.result, d_result, n * arr_size * sizeof(int),
                        hipMemcpyDeviceToHost));

    // --- print host output ---
    printf("[C] AFTER KERNEL: result[i][j] = j * m[i]\n");
    for (int i = 0; i < n; ++i) {
        printf("  inner[%d] m=%d: ", i, outer.inners[i].m);
        for (int j = 0; j < arr_size; ++j)
            printf("%3d ", outer.result[i * arr_size + j]);
        printf("\n");
    }

    HIP_CHECK(hipFree(d_inners));
    HIP_CHECK(hipFree(d_result));
}