/* particles/mod_epf_evolution/epf_evolution_device.h -- The EPF kernel launcher. */
#ifndef JOREK_EPF_EVOLUTION_DEVICE_H
#define JOREK_EPF_EVOLUTION_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "jgx/data/record_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * evolve_epf on the device: pack, upload, launch, pull, download, free.
 *
 * @param *_base, *_desc, n_*  the three record arrays with the registration that describes their layout
 * @param interp_base, interp_desc  the one jorek_fields_interp_linear record
 * @param rhs_data   the host feedback array, added into (not overwritten)
 * @param rhs_ext    its five extents, in Fortran declaration order
 * @param idx        the seven projection indices, 0-based, in epf_var order
 * @param proj_period  collect a projection every this many steps
 * @param[out] not_found, nf_R, nf_Z, bad_i_from, bad_i_to  the diagnostics the
 *                   kernel cannot print
 */
void jgx_device_epf_evolution_evolve_epf(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep,
    int32_t proj_period, double phi_search,
    int32_t* not_found, double* nf_R, double* nf_Z,
    int32_t* bad_i_from, int32_t* bad_i_to);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JOREK_EPF_EVOLUTION_DEVICE_H */
