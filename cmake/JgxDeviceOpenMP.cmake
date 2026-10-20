# JgxDeviceOpenMP.cmake -- the JGX device backend over OpenMP target offload.
#
# The device translation units are compiled by a second C++ compiler the
# project's languages cannot hold (nvc++ for NVIDIA, ROCm's clang++ for AMD), so
# they are built with custom commands into one shared library.
#
# NVIDIA: the OpenMP runtime is linked statically and kept internal: a shared
# libnvomp would interpose on the host's OpenMP runtime, and the device queries
# (e.g. the host device number) would be answered by the wrong one.
#
# AMD: ROCm ships libomp and libomptarget as shared libraries only, so they
# cannot be sealed in. The library carries a RUNPATH to ROCm's LLVM lib
# directory instead, and is exported with -Bsymbolic.
#
# Inputs:  JGX_DEVICE_ARCH (sm_NN or gfxNNNN), the jorek_common usage requirements.
# Outputs: the jgx_device target.

if(JGX_DEVICE_ARCH MATCHES "^sm_([0-9]+)$")
  find_program(JGX_OPENMP_CXX_COMPILER nvc++ DOC "The OpenMP offload C++ compiler")
  if(NOT JGX_OPENMP_CXX_COMPILER)
    message(FATAL_ERROR "JGX_DEVICE=openmp needs nvc++ on PATH (load the nvhpc module).")
  endif()
  set(_jgx_omp_gpu "-gpu=cc${CMAKE_MATCH_1}")

  set(JGX_OPENMP_COMPILE_FLAGS -std=c++20 -O3 -fPIC -mp=gpu ${_jgx_omp_gpu}
    # Floating-point sums in source order, as the other backends compute them.
    -Kieee -Mnovect)
  set(JGX_OPENMP_LINK_FLAGS -shared -mp=gpu ${_jgx_omp_gpu}
    -static-nvidia -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)

elseif(JGX_DEVICE_ARCH MATCHES "^gfx[0-9a-z]+$")
  find_program(JGX_OPENMP_CXX_COMPILER clang++
    HINTS "$ENV{ROCM_PATH}/lib/llvm/bin" NO_DEFAULT_PATH
    DOC "The OpenMP offload C++ compiler")
  if(NOT JGX_OPENMP_CXX_COMPILER)
    message(FATAL_ERROR
      "JGX_DEVICE=openmp on ${JGX_DEVICE_ARCH} needs ROCm's clang++ under "
      "$ROCM_PATH/lib/llvm/bin (load the rocm module).")
  endif()
  get_filename_component(_jgx_omp_llvm "${JGX_OPENMP_CXX_COMPILER}" DIRECTORY)
  get_filename_component(_jgx_omp_llvm "${_jgx_omp_llvm}" DIRECTORY)
  set(_jgx_omp_gpu "--offload-arch=${JGX_DEVICE_ARCH}")

  set(JGX_OPENMP_COMPILE_FLAGS -std=c++20 -O3 -fPIC -fopenmp ${_jgx_omp_gpu}
    # Contraction as clang applies it to the HIP backend's device code, and
    # sums in source order.
    -ffp-contract=fast-honor-pragmas -fno-vectorize -fno-slp-vectorize)
  set(JGX_OPENMP_LINK_FLAGS -shared -fopenmp ${_jgx_omp_gpu}
    "-Wl,-rpath,${_jgx_omp_llvm}/lib" -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)

else()
  message(FATAL_ERROR
    "JGX_DEVICE=openmp supports one NVIDIA sm_NN or AMD gfxNNNN architecture, "
    "not '${JGX_DEVICE_ARCH}'")
endif()

set(JGX_DEVICE_SOURCES
  "jgx/cpp/backends/openmp/jgx_backend_openmp.openmp.cpp"
  "jgx/cpp/backends/openmp/jgx_pack_openmp.openmp.cpp"
  "models/phys_module/phys_device.hip.cpp"
  "particles/mod_runaway_evolution/runaway_evolution_device.openmp.cpp")

# jorek_common's compile usage requirements only, as for the other backends.
set(_jgx_omp_defs
  "-D$<JOIN:$<TARGET_PROPERTY:jorek_common,INTERFACE_COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
  -DJGX_DEVICE_OPENMP)
set(_jgx_omp_incs
  "-I$<JOIN:$<TARGET_PROPERTY:jorek_common,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>")

set(_jgx_omp_objs "")
foreach(_src IN LISTS JGX_DEVICE_SOURCES)
  set(_obj "${CMAKE_CURRENT_BINARY_DIR}/jgx_device_openmp/${_src}.o")
  get_filename_component(_dir "${_obj}" DIRECTORY)
  file(MAKE_DIRECTORY "${_dir}")
  add_custom_command(
    OUTPUT "${_obj}"
    COMMAND "${JGX_OPENMP_CXX_COMPILER}" ${JGX_OPENMP_COMPILE_FLAGS}
            ${_jgx_omp_defs} ${_jgx_omp_incs}
            -MMD -MF "${_obj}.d" -c "${CMAKE_CURRENT_SOURCE_DIR}/${_src}" -o "${_obj}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}"
    DEPFILE "${_obj}.d"
    COMMENT "Building OpenMP offload object ${_src}"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  list(APPEND _jgx_omp_objs "${_obj}")
endforeach()

set(JGX_DEVICE_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/libjgx_device_openmp.so")
add_custom_command(
  OUTPUT "${JGX_DEVICE_LIBRARY}"
  COMMAND "${JGX_OPENMP_CXX_COMPILER}" ${JGX_OPENMP_LINK_FLAGS}
          ${_jgx_omp_objs} -o "${JGX_DEVICE_LIBRARY}"
  DEPENDS ${_jgx_omp_objs}
  COMMENT "Linking the JGX OpenMP offload library (${JGX_DEVICE_ARCH})"
  VERBATIM)
add_custom_target(jgx_device_link DEPENDS "${JGX_DEVICE_LIBRARY}")

add_library(jgx_device INTERFACE)
add_dependencies(jgx_device jgx_device_link)
target_link_libraries(jgx_device INTERFACE "${JGX_DEVICE_LIBRARY}")
target_link_options(jgx_device INTERFACE "LINKER:-rpath,${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "JGX device backend: openmp, ${JGX_OPENMP_CXX_COMPILER}, arch ${JGX_DEVICE_ARCH}")
