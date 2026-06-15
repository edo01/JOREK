#define USE_GPU 1
#define BLOCK_SIZE 256
#define GPU_DEBUG 0

#define ORDERING_TYPE 0
#define NODES_FIRST       0
#define ELEMENTS_FIRST    0
#define FB_ELEMENTS_FIRST 1

#define STEPS_PER_BATCH 2
#define FB_LANE_FANOUT 8

#define LUT_VALUES_DELTAS 0
#define LUT_DEBUG            0
#define LUT_N_SLOTS          8
#define LUT_NEIGHBOR_PRELOAD 1