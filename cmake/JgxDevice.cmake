# JgxDevice.cmake -- the JGX device backend.
#
# Defines jgx_device, a static library holding the .hip.cpp translation units,
# which jorek_core links publicly. Nothing here runs when JGX_DEVICE=off.
#
# The sources are HIP for both vendors; only the build differs. JGX_DEVICE_ARCH
# selects the branch:
#
#   sm_NN    NVIDIA. Compiled as CUDA: HIP-on-NVIDIA is a header layer over
#            CUDA, so hipcc would exec nvcc in any case.
#   gfxNNNN  AMD. Compiled as HIP.
#
# Inputs:  JGX_DEVICE, JGX_DEVICE_ARCH, the jorek_common usage requirements.
# Outputs: the jgx_device target, and JGX_HAS_DEVICE on jorek_common.

if(JGX_DEVICE STREQUAL "off")
  return()
endif()

# Host-visible: "a device backend is present"
target_compile_definitions(jorek_common INTERFACE JGX_HAS_DEVICE)

# --- Toolchain --------------------------------------------------------------
#
# hip-config.cmake is preferred; ROCM_PATH is the fallback for installs that do
# not place lib/cmake on CMAKE_PREFIX_PATH.
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

  # Located explicitly: the build drives nvcc or clang++ directly, and hipcc,
  # which would supply this path itself, is not used.
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

# --- Architecture -----------------------------------------------------------
#
# HIP sources are architecture-neutral; only the build is told. The two
# spellings are separated here so that each branch deals in one of them, and so
# that an unrecognised entry fails at configure time rather than yielding a
# binary no device will run.
if(NOT JGX_DEVICE_ARCH)
  message(FATAL_ERROR
    "JGX_DEVICE=${JGX_DEVICE} needs JGX_DEVICE_ARCH, which has no default -- it is a "
    "property of the machine. Use a preset that sets it (e.g. sm_90, gfx942) or pass "
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

# --- Sources ----------------------------------------------------------------
#
# Listed explicitly rather than globbed: JorekSources.cmake excludes *.hip.cpp
# from its per-directory glob, so a device translation unit may sit beside the
# host sources of its module without reaching the host compiler.
set(JGX_DEVICE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/jgx/cpp/backends/hip/jgx_backend_hip.hip.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/jgx/cpp/backends/hip/jgx_pack_hip.hip.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/phys_module/phys_device.hip.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/particles/mod_runaway_evolution/runaway_evolution_device.hip.cpp")

if(JGX_DEVICE_AMD_ARCH)

  # --- AMD ------------------------------------------------------------------
  #
  # Device code must be relocatable. phys_device.hip.cpp defines d_phys, which
  # runaway_evolution_device.hip.cpp reads; without -fgpu-rdc the amdgcn link
  # rejects that cross-unit reference while the reading unit is still being
  # compiled. This is the AMD equivalent of CUDA_SEPARABLE_COMPILATION.
  #
  # CMake offers no HIP counterpart to CUDA_SEPARABLE_COMPILATION or
  # CUDA_RESOLVE_DEVICE_SYMBOLS -- no such HIP_* property exists as of 4.2 --
  # and the final link is performed by ifx, which cannot device-link. The step
  # is therefore issued explicitly over an OBJECT library:
  #
  #   <hip compiler> --hip-link -fgpu-rdc --offload-arch=<arch> -r <objects>
  #
  # -r is required. Without it the driver emits a bare fat binary, which a
  # foreign linker cannot consume; with it the driver performs the device link,
  # wraps the resulting image and its registration code into an object, and
  # partial-links that with the host objects. The result is one ELF relocatable,
  # which enters jgx_device as an external object.
  enable_language(HIP)

  if(NOT hip_FOUND)
    message(FATAL_ERROR
      "JGX_DEVICE_ARCH='${JGX_DEVICE_ARCH}' is an AMD target, which needs the hip::host "
      "imported target for the runtime on the executables' link line, but "
      "find_package(hip CONFIG) did not find hip-config.cmake. Load the ROCm module "
      "(it puts <rocm>/lib/cmake on CMAKE_PREFIX_PATH) rather than only setting ROCM_PATH.")
  endif()

  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)

  set_source_files_properties(${JGX_DEVICE_SOURCES} PROPERTIES LANGUAGE HIP)

  add_library(jgx_device_objs OBJECT ${JGX_DEVICE_SOURCES})

  # jorek_common's compile usage requirements only. Linking it would also place
  # JOREK_LIBRARIES on the device-link line, which does not accept the host
  # toolchain's flags; device units link nothing and are reached via extern "C".
  target_include_directories(jgx_device_objs PRIVATE
    $<TARGET_PROPERTY:jorek_common,INTERFACE_INCLUDE_DIRECTORIES>)

  target_compile_definitions(jgx_device_objs PRIVATE
    $<TARGET_PROPERTY:jorek_common,INTERFACE_COMPILE_DEFINITIONS>
    JGX_DEVICE_HIP)                # JGX_HD -> __host__ __device__ (jgx/macros.h)
  # __HIP_PLATFORM_AMD__ needs no counterpart to the NVIDIA branch's define: the
  # driver sets it when compiling HIP as HIP.

  # Required; see above.
  target_compile_options(jgx_device_objs PRIVATE -fgpu-rdc)
  # -Werror=cross-execution-space-call likewise has no counterpart: clang already
  # rejects a host call from a __host__ __device__ function as an error.

  set_target_properties(jgx_device_objs PROPERTIES
    HIP_ARCHITECTURES "${JGX_DEVICE_AMD_ARCH}")

  set(JGX_DEVICE_OFFLOAD_FLAGS "")
  foreach(_arch IN LISTS JGX_DEVICE_AMD_ARCH)
    list(APPEND JGX_DEVICE_OFFLOAD_FLAGS "--offload-arch=${_arch}")
  endforeach()

  set(JGX_DEVICE_LINK_OBJ "${CMAKE_CURRENT_BINARY_DIR}/jgx_device_link.o")
  add_custom_command(
    OUTPUT "${JGX_DEVICE_LINK_OBJ}"
    COMMAND "${CMAKE_HIP_COMPILER}" --hip-link -fgpu-rdc ${JGX_DEVICE_OFFLOAD_FLAGS}
            -r $<TARGET_OBJECTS:jgx_device_objs> -o "${JGX_DEVICE_LINK_OBJ}"
    DEPENDS jgx_device_objs $<TARGET_OBJECTS:jgx_device_objs>
    COMMENT "Device-linking the JGX HIP objects (${JGX_DEVICE_AMD_ARCH})"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  add_custom_target(jgx_device_link DEPENDS "${JGX_DEVICE_LINK_OBJ}")

  set_source_files_properties("${JGX_DEVICE_LINK_OBJ}" PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE)

  add_library(jgx_device STATIC "${JGX_DEVICE_LINK_OBJ}")
  add_dependencies(jgx_device jgx_device_link)
  # The archive holds one pre-linked object and no source of a known language.
  set_target_properties(jgx_device PROPERTIES LINKER_LANGUAGE CXX)

  # The HIP runtime does not arrive with the Fortran driver. PUBLIC, so that it
  # reaches every executable pulling jgx_device in through jorek_core.
  target_link_libraries(jgx_device PUBLIC hip::host)

else()

  # --- NVIDIA ---------------------------------------------------------------
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set_source_files_properties(${JGX_DEVICE_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_library(jgx_device STATIC ${JGX_DEVICE_SOURCES})

  # jorek_common's compile usage requirements only. Linking it would also place
  # JOREK_LIBRARIES on the link line, and `nvcc -dlink` passes its link flags to
  # gcc, which knows neither -fiopenmp nor MKL's.
  target_include_directories(jgx_device PRIVATE
    $<TARGET_PROPERTY:jorek_common,INTERFACE_INCLUDE_DIRECTORIES>
    "${JGX_HIP_INCLUDE_DIR}")

  target_compile_definitions(jgx_device PRIVATE
    $<TARGET_PROPERTY:jorek_common,INTERFACE_COMPILE_DEFINITIONS>
    JGX_DEVICE_HIP                 # JGX_HD -> __host__ __device__ (jgx/macros.h)
    __HIP_PLATFORM_NVIDIA__)       # hipcc sets this; driving nvcc, we must

  target_compile_options(jgx_device PRIVATE
    # Required. nvcc reports a host call from a __host__ __device__ function as
    # a warning: the object builds and the device link succeeds, so reading host
    # state from a kernel would surface first as incorrect physics.
    -Werror=cross-execution-space-call
    # Silences the HIP-on-NVIDIA headers' diagnostics about themselves.
    -Wno-deprecated-declarations
    --diag-suppress=174)

  set_target_properties(jgx_device PROPERTIES
    CUDA_ARCHITECTURES "${JGX_DEVICE_CUDA_ARCH}"   # sm_90 -> 90, list-for-list
    # d_phys is read across translation units, so device code must be
    # relocatable; and the final link is performed by ifx, which cannot
    # device-link, so `nvcc -dlink` runs here, on the archive.
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # The CUDA runtime does not arrive with the Fortran driver. PUBLIC, so that it
  # reaches every executable pulling jgx_device in through jorek_core.
  target_link_libraries(jgx_device PUBLIC CUDA::cudart)

endif()

message(STATUS "JGX device backend: ${JGX_DEVICE}, arch ${JGX_DEVICE_ARCH}, "
               "HIP headers ${JGX_HIP_INCLUDE_DIR}")
