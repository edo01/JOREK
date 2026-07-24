/*
 * USE_GPU:               0 = CPU-only build, 1 = enable GPU (HIP) code paths.
 * SORTING_BLOCK_SIZE:    GPU threads per block for the counting-sort helper kernels
 *                        (count/scan/scatter), shared by both code paths. USE_GPU=1 only.
 * GPU_DEBUG:             0 = off, 1 = verbose GPU debug output + hipDeviceSynchronize after each launch. USE_GPU=1 only.
 *
 * ORDERING_TYPE:         Particle ordering before the coupling scheme loop, Fortran-side.      <-- Old optimization from Edoardo's code
 *                        0 = none, 1 = global sort, 2 = SIMD-lane sort.
 * NODES_FIRST:           Memory layout of nl_x/nl_values/nl_deltas.
 *                        0 = n_nodes last, 1 = n_nodes first.
 * ELEMENTS_FIRST:        Memory layout of el_vertex/el_neighbours/el_size.
 *                        0 = n_elements last, 1 = n_elements first.
 * FB_ELEMENTS_FIRST:     Memory layout of feedback_rhs buffer.
 *                        0 = (..., n_elements), more cache-friendly when particle globally ordered,
 *                        1 = (n_elements,[FB_LANE_FANOUT,]...) improves GPU warp coalescing.
 *
 * USE_BATCH_KERNEL:      0 = split-kernel (default): proj_stage + proj_accumulate (atomic-free) + evolve_push,
 *                            overlapped on two HIP streams.
 *                        1 = batch-kernel: fused evolve_batch_kernel runs STEPS_PER_BATCH steps in one launch,
 *                            particle state held in registers across steps; optional shared-memory LUT (LUT_*).
 *
 * Split-kernel knobs (USE_BATCH_KERNEL=0):
 * SP_BLOCK_SIZE:         Threads per block for proj_stage_kernel and evolve_push_kernel.
 *                        Their __launch_bounds__ min-blocks hint is derived so the
 *                        guaranteed resident thread count per SM stays at 512 regardless
 *                        of the swept value (keeps the per-thread register budget fixed).
 * PARTICLES_PER_THREAD:  Particles processed by each thread of proj_stage_kernel and
 *                        evolve_push_kernel via a grid-stride loop (grid is shrunk by this
 *                        factor). 1 = one thread per particle (previous behaviour).
 *                        Split-kernel only; the batch kernel is always one thread per particle.
 * PROJ_TILE:             Particles per tile in proj_accumulate_kernel.
 * ACCUM_BLOCK_SIZE:      Threads per block for proj_accumulate_kernel.
 * ACCUM_MIN_BLOCKS_PER_SM: __launch_bounds__ min-blocks hint for proj_accumulate (0 = omit hint).
 *
 * Batch-kernel knobs (USE_BATCH_KERNEL=1):
 * BATCH_BLOCK_SIZE:      Threads per block for evolve_batch_kernel.
 * STEPS_PER_BATCH:       Kinetic steps per kernel launch; particles are re-sorted by element between batches.
 *                        0 = all steps in one launch with no intermediate sort. ORDERING_TYPE>0 only.
 * FB_LANE_FANOUT:        Per-lane feedback_rhs replicas to reduce atomic contention.
 *
 * LUT knobs (USE_BATCH_KERNEL=1):
 * LUT_VALUES_DELTAS:     0 = off, 1 = cache nl_values/nl_deltas in shared memory per block.
 *                        Built once per batch from the counting-sorted order; requires batch sorting enabled.
 * LUT_N_SLOTS:           Cache slots (distinct elements) per block. Cost: LUT_N_SLOTS*512*N_TOR bytes of shared mem.
 *                        A static_assert guards the 2-blocks/CU occupancy budget.
 * LUT_NEIGHBOR_PRELOAD:  0 = cache base (step-0) elements only; 1 = also fill leftover slots with mesh neighbours
 *                        to capture step-1 particle drift. Boundary edges (neighbour=0) are skipped.
 * LUT_DEBUG:             0 = off, 1 = print hit/miss counters per batch (PROJ + PUSH).
 *
 * CCOLL_DUMP:            0 = off, 1 = dump per-particle small-angle-collision uin/uout for
 *                        CPU-vs-GPU statistical validation (benchmarks/small_angle_collision/
 *                        compare_ccoll_moments.py). CPU writes CCOLL_DUMP_CPU_FILE; GPU writes
 *                        CCOLL_DUMP_GPU_FILE".rank<id>". Debug/validation only -- keep 0 for production.
 */

#define USE_GPU 1
#define SORTING_BLOCK_SIZE 256
#define GPU_DEBUG 0

#define ORDERING_TYPE 0
#define NODES_FIRST       0
#define ELEMENTS_FIRST    0
#define FB_ELEMENTS_FIRST 0

#define USE_BATCH_KERNEL 0

/* SPECIALIZED KERNELS */
#define SP_BLOCK_SIZE 256
#define S_MIN_BLOCKS_PER_CU 2
#define PARTICLES_PER_THREAD 1
#define PROJ_TILE 256
#define ACCUM_BLOCK_SIZE 384
#define ACCUM_MIN_BLOCKS_PER_SM 2

/* BATCH KERNEL */
#define BATCH_BLOCK_SIZE 256
#define STEPS_PER_BATCH 2
#define FB_LANE_FANOUT 8

#define LUT_VALUES_DELTAS 0
#define LUT_DEBUG            0
#define LUT_N_SLOTS          8
#define LUT_NEIGHBOR_PRELOAD 1

/* CCOLL VALIDATION DUMP (debug only) */
#define CCOLL_DUMP 0
#define CCOLL_DUMP_CPU_FILE "ccoll_du.cpu"
#define CCOLL_DUMP_GPU_FILE "ccoll_du.gpu"
