/* models/phys_module/phys_device.hip.cpp -- the device mirror of the pushed copy
 * of phys_module's scalars.
 *
 * Storage only, the same role phys.cpp plays for the host: Fortran fills it
 * through jgx_c_set_phys, and everything downstream reads it through
 * jorek::phys(), which resolves to d_phys in device code.
 *
 * Why a device global and not a jgx_buf: a buffer's device pointer has to arrive
 * as a kernel argument, and phys is read by leaves that take no such argument --
 * threading it through every signature is exactly what the pushed copy exists to
 * avoid. See the device-globals note in jgx/jgx_c_api.h.
 *
 * Why the file is here and not in the backend: this is the half that knows what
 * a phys_state is, and jgx/ names no JOREK type. It is also the half that names
 * no backend -- there is not a hip* call in it. The device language arrives
 * through jgx/device.h and every transfer through jgx_c_*, so the same source
 * serves whatever backend jgx grows next.
 */
#include "models/phys_module/phys.h"

#include "jgx/device.h"
#include "jgx/jgx_c_api.h"

namespace jorek {

/* Declared in phys.h, because phys() returns it. */
JGX_DEVICE_VAR phys_state d_phys;

void phys_push_to_device(const phys_state& p) {
  jgx_c_push_symbol(JGX_SYMBOL(d_phys), &p, sizeof(phys_state));
}

void phys_pull_from_device(phys_state& out) {
  jgx_c_pull_symbol(&out, JGX_SYMBOL(d_phys), sizeof(phys_state));
}

} /* namespace jorek */
