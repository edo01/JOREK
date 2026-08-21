/* models/phys_module/phys_device_off.cpp -- phys.h's two transfers when there is
 * no device backend.
 *
 * They are declared unconditionally so that jgx_c_set_phys, and anything else
 * that pushes, carries no backend guard. With no device THIS FILE IS NOT COMPILED
 * and the right phys_device.*.cpp with the right definition is compiled instead.
 */
#include "models/phys_module/phys.h"

#ifndef JGX_HAS_DEVICE

namespace jorek {

void phys_push_to_device(const phys_state&) {}
void phys_pull_from_device(phys_state&) {}

} /* namespace jorek */

#endif /* !JGX_HAS_DEVICE */
