/* particles/mod_runaway_evolution/runaway_evolution_device.h -- The REs kernel launcher. */
#ifndef JOREK_RUNAWAY_EVOLUTION_DEVICE_H
#define JOREK_RUNAWAY_EVOLUTION_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "jgx/data/record_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * evolve_REs on the device: pack, upload, launch, pull, download, free.
 *
 * @param *_base, *_desc, n_*  the three record arrays as Fortran holds them,
 *                             with the registration that describes their layout
 * @param interp_base, interp_desc  the one jorek_fields_interp_linear record
 * @param rhs_data   the host feedback array, added into (not overwritten)
 * @param rhs_ext    its five extents, in Fortran declaration order
 * @param idx        the three projection indices, 0-based
 * @param[out] not_found, nf_R, nf_Z, bad_i_from, bad_i_to  the diagnostics the
 *                   kernel cannot print
 */
void jgx_device_runaway_evolution_evolve_REs(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep, double phi_search,
    int32_t* not_found, double* nf_R, double* nf_Z,
    int32_t* bad_i_from, int32_t* bad_i_to);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JOREK_RUNAWAY_EVOLUTION_DEVICE_H */
