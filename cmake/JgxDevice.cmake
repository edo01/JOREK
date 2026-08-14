# The JGX device backend: one static library, jgx_device, holding the .hip.cpp
# translation units. Nothing here runs when JGX_DEVICE=off.
#
# The sources are HIP, the language CMake compiles them as is CUDA (TODO: 
# AMD BRANCH).

if(JGX_DEVICE STREQUAL "off")
  return()
endif()

# --- Toolchain --------------------------------------------------------------
find_package(hip CONFIG QUIET)
if(hip_FOUND)
  set(JGX_HIP_INCLUDE_DIR "${hip_INCLUDE_DIR}")
  message(STATUS "JGX device backend: HIP ${hip_VERSION} found via find_package(hip), "
                 "ROCM_PATH not needed")
else()
  if(NOT DEFINED ENV{ROCM_PATH})
    message(FATAL_ERROR
      "JGX_DEVICE=${JGX_DEVICE} needs the HIP install in the environment (no hip-config.cmake "
      "on CMAKE_PREFIX_PATH either). Provide ROCM_PATH.")
  endif()

  # hipcc does not add its own include path, and we are not using hipcc anyway.
  find_path(JGX_HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    HINTS "$ENV{ROCM_PATH}/include"
    DOC "Directory holding hip/hip_runtime.h")
  if(NOT JGX_HIP_INCLUDE_DIR)
    message(FATAL_ERROR
      "hip/hip_runtime.h not found under $ENV{ROCM_PATH}/include -- is ROCM_PATH the "
      "rocm-<version> prefix rather than the install root?")
  endif()
endif()

# --- Which architecture, and therefore which host toolchain -----------------
#
# HIP source is architecture-neutral; only the build has to be told. The two
# spellings are sorted here so that everything below deals in one of them, and so
# that a wrong entry is a configure error rather than a card the binary silently
# will not run on.
if(NOT JGX_DEVICE_ARCH)
  message(FATAL_ERROR
    "JGX_DEVICE=${JGX_DEVICE} needs JGX_DEVICE_ARCH, which has no default -- it is a "
    "property of the machine. Use a preset that sets it (pitagora-hip: sm_90) or pass "
    "-DJGX_DEVICE_ARCH=<arch>.")
endif()

set(JGX_DEVICE_CUDA_ARCH "")
set(JGX_DEVICE_AMD_ARCH "")
foreach(_arch IN LISTS JGX_DEVICE_ARCH)
  if(_arch MATCHES "^sm_([0-9]+[a-z]?)$")
    list(APPEND JGX_DEVICE_CUDA_ARCH "${CMAKE_MATCH_1}")
  elseif(_arch MATCHES "^gfx[0-9a-z]+$")
    list(APPEND JGX_DEVICE_AMD_ARCH "${_arch}")
  else()
    message(FATAL_ERROR
      "JGX_DEVICE_ARCH entry '${_arch}' is neither an NVIDIA sm_NN nor an AMD gfxNNNN")
  endif()
endforeach()

if(JGX_DEVICE_CUDA_ARCH AND JGX_DEVICE_AMD_ARCH)
  message(FATAL_ERROR
    "JGX_DEVICE_ARCH mixes NVIDIA and AMD targets ('${JGX_DEVICE_ARCH}'); one build "
    "produces code for one vendor.")
endif()

#@TODO: AMD arm is not written
if(JGX_DEVICE_AMD_ARCH)
  message(FATAL_ERROR
    "JGX_DEVICE_ARCH='${JGX_DEVICE_ARCH}' is an AMD target. The sources are HIP and need "
    "no change, but this file only implements the NVIDIA route (enable_language(CUDA), "
    "because hipcc on this machine is a wrapper over nvcc). An AMD host wants "
    "enable_language(HIP) with CMAKE_HIP_ARCHITECTURES -- the ROCm install even ships "
    "lib/cmake/hip-lang -- and that arm has never been compiled, so it is not here.")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- The library ------------------------------------------------------------
#
# Listed explicitly, not globbed: JorekSources.cmake drops every *.hip.cpp from
# its per-directory glob, so a device translation unit can sit next to the host
# sources of the module it belongs to without icpx ever seeing it.
set(JGX_DEVICE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/jgx/cpp/backends/hip/jgx_backend_hip.hip.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/phys_module/phys_device.hip.cpp")

set_source_files_properties(${JGX_DEVICE_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(jgx_device STATIC ${JGX_DEVICE_SOURCES})

# jorek_common's compile usage requirements -- the model defines the ported
# kernels size their scratch with, and the include paths -- but deliberately NOT
# by linking it: that would also put JOREK_LIBRARIES on the link line, and the
# `nvcc -dlink` step below hands its link flags to gcc, which does not know
# -fiopenmp or MKL's. Nothing in a device TU needs to link anyway; it is reached
# through extern "C".
target_include_directories(jgx_device PRIVATE
  $<TARGET_PROPERTY:jorek_common,INTERFACE_INCLUDE_DIRECTORIES>
  "${JGX_HIP_INCLUDE_DIR}")

target_compile_definitions(jgx_device PRIVATE
  $<TARGET_PROPERTY:jorek_common,INTERFACE_COMPILE_DEFINITIONS>
  JGX_DEVICE_HIP                 # JGX_HD -> __host__ __device__ (jgx/macros.h)
  __HIP_PLATFORM_NVIDIA__)       # hipcc sets this; driving nvcc, we must

target_compile_options(jgx_device PRIVATE
  # Not optional. nvcc reports calling a host function from a __host__ __device__
  # one as a *warning*: the object builds, the device link succeeds, and reading
  # host state from a kernel would first appear as wrong physics. Measured on the
  # evolve_RE chain -- five hits, all jorek::phys(), and nothing else.
  -Werror=cross-execution-space-call
  # The HIP-on-NVIDIA headers emit ~200 lines about themselves per TU.
  -Wno-deprecated-declarations
  --diag-suppress=174)

set_target_properties(jgx_device PROPERTIES
  CUDA_ARCHITECTURES "${JGX_DEVICE_CUDA_ARCH}"   # sm_90 -> 90, list-for-list
  # The mirror of the phys copy is an `extern __device__` variable read from
  # another translation unit, so device code has to be relocatable. And the
  # final link is done by ifx, which cannot device-link -- so nvcc -dlink must
  # run here, on the archive, rather than at the executable.
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The CUDA runtime does not arrive with the Fortran driver. PUBLIC, so it
# reaches every executable that pulls jgx_device in through jorek_core.
target_link_libraries(jgx_device PUBLIC CUDA::cudart)

message(STATUS "JGX device backend: ${JGX_DEVICE}, arch ${JGX_DEVICE_ARCH}, "
               "HIP headers ${JGX_HIP_INCLUDE_DIR}")
