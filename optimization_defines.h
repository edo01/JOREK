#define USE_GPU 1
#define BLOCK_SIZE 256
#define GPU_DEBUG 0

#define ORDERING_TYPE 0
#define NODES_FIRST       0
#define ELEMENTS_FIRST    0
#define FB_ELEMENTS_FIRST 0

/* Kernel strategy selector: comment out to use split-kernel (default), uncomment for batch-kernel */
/* #define USE_BATCH_KERNEL */

/* Split-kernel strategy knobs (kernel_re_evolution_split.hip.cpp) */
#define PROJ_TILE 64
#define ACCUM_BLOCK_SIZE 256
#define ACCUM_MIN_BLOCKS_PER_SM 0

/* Batch-kernel strategy knobs (kernel_re_evolution_batch.hip.cpp) */
#define STEPS_PER_BATCH 2
#define FB_LANE_FANOUT 8

#define LUT_VALUES_DELTAS 0
#define LUT_DEBUG            0
#define LUT_N_SLOTS          8
#define LUT_NEIGHBOR_PRELOAD 1
