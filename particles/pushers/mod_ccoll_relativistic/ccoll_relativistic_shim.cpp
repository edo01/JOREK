/* The C entry points of ccoll_relativistic.h.
 *
 * Only the two routines that take no field set are here.  The wrapper
 * ccoll_kinetic_relativistic_push cannot have one for the same reason
 * type_fields' own methods cannot: it needs c_loc of the interpolator, which is
 * a polymorphic allocatable, so whoever already knows the strategy has to do
 * the select type -- today that is evolve_REs.  See particles/mod_fields/.
 */
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"

#include <cstddef>
#include <cstdint>

namespace {

/* The table arrives as the four arrays and their two extents, as it does
 * everywhere: ccoll_data is not a registered record. */
ccoll::ccoll_table<double> table_from(std::int32_t nu, std::int32_t nth,
                                      const double* u, const double* theta,
                                      const double* L0, const double* L1,
                                      double mi, double Z0) {
  return ccoll::make_ccoll_table<double>(static_cast<int>(nu), static_cast<int>(nth),
                                         u, theta, L0, L1, mi, Z0);
}

} /* anonymous namespace */

extern "C" {

/* mod_ccoll_relativistic::interp_L0L1 */
void jgx_host_ccoll_interp_L0L1(const std::int32_t nu, const std::int32_t nth,
                                const double* u_tab, const double* theta_tab,
                                const double* L0_tab, const double* L1_tab,
                                const double u, const double theta,
                                double* L0, double* L1) {
  const auto dat = table_from(nu, nth, u_tab, theta_tab, L0_tab, L1_tab, 0.0, 0.0);
  ccoll::interp_L0L1(dat, u, theta, *L0, *L1);
}

/* mod_ccoll_relativistic::ccoll_kinetic_relativistic_explicitpush */
void jgx_host_ccoll_kinetic_relativistic_explicitpush(
    const std::int32_t nu, const std::int32_t nth,
    const double* u_tab, const double* theta_tab,
    const double* L0_tab, const double* L1_tab,
    const double mi, const double Z0,
    const double ma, const std::int32_t qa,
    const double ne, const double the, const double ni, const double thi,
    const double dt, const double* rnd, const double* uin, double* uout) {
  const auto dat = table_from(nu, nth, u_tab, theta_tab, L0_tab, L1_tab, mi, Z0);
  ccoll::ccoll_kinetic_relativistic_explicitpush(
      dat, ma, static_cast<int>(qa), ne, the, ni, thi, dt, rnd, uin, uout);
}

} /* extern "C" */
