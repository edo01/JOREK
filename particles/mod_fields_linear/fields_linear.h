/* particles/mod_fields_linear/fields_linear.h -- the linearly interpolated
 * field interpolator.
 *
 * Mirrors jorek_fields_interp_linear
 * (particles/mod_fields_linear/mod_fields_linear.f90): the extension of
 * fields_base is C++ inheritance, its type-bound procedures are member
 * functions carrying the binding name.
 */
#ifndef FIELDS_LINEAR_H
#define FIELDS_LINEAR_H

#include <cmath>
#include <cstddef>

#include "elements/mod_interp/interp.h"
#include "particles/mod_fields/fields_set.h"
#include "models/phys_module/phys.h"
#include "models/mod_settings/mod_settings.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace jorek {

/* What jorek_fields_interp_linear adds to fields_interpolator, continuing the
 * shared numbering. Mirrors mod_jgx_fields_interp_linear_record.f90. */
enum fields_interp_linear_field {
  JGX_FIL_TIME_NOW = JGX_FI_BASE_COUNT,
  JGX_FIL_TIME_PREV,
  JGX_FIL_COUNT
};

template <class L, class Real = double, class Int = int>
struct fields_interp_linear_set : fields_base_set<L, Real, Int> {
  using base = fields_base_set<L, Real, Int>;

  /* The base is dependent; unqualified lookup needs these. */
  using base::element_list;
  using base::node_list;
  using base::is_static;

  Real time_now  = 0;  /* SI */
  Real time_prev = 0;  /* SI */

  /**
   * jorek_fields_interp_linear%interp_PRZ (do_interp_PRZ_1) -- interpolate n_v
   * variables and the geometry at (s,t,phi) in element ie, then interpolate
   * linearly in time between the two restart files the field set holds.
   *
   * @param time      time to interpolate at, SI
   * @param ie        element index into element_list
   * @param i_v       indices of the n_v variables to interpolate
   * @param n_v       number of variables, <= JGX_N_VALUES_MAX
   * @param s, t      element-local coordinates
   * @param phi       toroidal angle
   * @param[out] P, P_s, P_t, P_phi, P_time  variables, their s/t/phi derivatives
   *                  and their time derivative (size n_v)
   * @param[out] R, R_s, R_t, Z, Z_s, Z_t    geometry and its derivatives
   */
  template<class IdxView, class OutView>
  JGX_HD void interp_PRZ(const double time, const std::size_t ie,
                         const IdxView& i_v, const int n_v,
                         const double s, const double t, const double phi,
                         OutView P, OutView P_s, OutView P_t,
                         OutView P_phi, OutView P_time,
                         double& R, double& R_s, double& R_t,
                         double& Z, double& Z_s, double& Z_t) const {
    for (int i = 0; i < n_v; ++i) P_time(i) = 0.0;

    const double t_jorek = phys().t_jorek;

    interp::interp_PRZ_1(element_list, node_list, ie, i_v, n_v, s, t, phi,
                         false, P, P_s, P_t, P_phi,
                         R, R_s, R_t, Z, Z_s, Z_t);

    if (t_jorek <= 0.0) return;

    // The deltas, in the scratch the Fortran gets from automatic arrays.
    const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
    Real Pd_[JGX_N_VALUES_MAX], Pd_s_[JGX_N_VALUES_MAX];
    Real Pd_t_[JGX_N_VALUES_MAX], Pd_phi_[JGX_N_VALUES_MAX];
    const jgx::view<Real, 1> Pd(Pd_, pe), Pd_s(Pd_s_, pe);
    const jgx::view<Real, 1> Pd_t(Pd_t_, pe), Pd_phi(Pd_phi_, pe);

    // R..Z_t come out identical: the geometry does not read values/deltas.
    interp::interp_PRZ_1(element_list, node_list, ie, i_v, n_v, s, t, phi,
                         true, Pd, Pd_s, Pd_t, Pd_phi,
                         R, R_s, R_t, Z, Z_s, Z_t);

    double dt;
    if (fabs(time_now - time_prev) > 1e-10 && !is_static) {
      dt = 1.0/(time_now - time_prev);
      const double df = (time_now - time)*dt;
      for (int i = 0; i < n_v; ++i) {
        P    (i) -= df*Pd    (i);
        P_s  (i) -= df*Pd_s  (i);
        P_t  (i) -= df*Pd_t  (i);
        P_phi(i) -= df*Pd_phi(i);
      }
    } else {
      dt = 1.0/t_jorek;
    }

    for (int i = 0; i < n_v; ++i) P_time(i) = Pd(i)*dt;
  } // interp_PRZ

  /**
   * jorek_fields_interp_linear%interp_PRZP_1 (do_interp_PRZP_1) -- as
   * interp_PRZ, for a non-axisymmetric configuration: the geometry has phi
   * derivatives of its own.
   *
   * @param time            time to interpolate at, SI
   * @param ie              element index into element_list
   * @param i_v             indices of the n_v variables to interpolate
   * @param n_v             number of variables, <= JGX_N_VALUES_MAX
   * @param s, t            element-local coordinates
   * @param phi             toroidal angle
   * @param[out] P, P_s, P_t, P_phi, P_time  variables, their s/t/phi derivatives
   *                        and their time derivative (size n_v)
   * @param[out] R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi  geometry and its derivatives
   */
  template<class IdxView, class OutView>
  JGX_HD void interp_PRZP_1(const double time, const std::size_t ie,
                            const IdxView& i_v, const int n_v,
                            const double s, const double t, const double phi,
                            OutView P, OutView P_s, OutView P_t,
                            OutView P_phi, OutView P_time,
                            double& R, double& R_s, double& R_t, double& R_phi,
                            double& Z, double& Z_s, double& Z_t, double& Z_phi) const {
    for (int i = 0; i < n_v; ++i) P_time(i) = 0.0;

    const double t_jorek = phys().t_jorek;

    interp::interp_PRZP_1(element_list, node_list, ie, i_v, n_v, s, t, phi,
                          false, P, P_s, P_t, P_phi,
                          R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi);

    if (t_jorek <= 0.0) return;

    const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
    Real Pd_[JGX_N_VALUES_MAX], Pd_s_[JGX_N_VALUES_MAX];
    Real Pd_t_[JGX_N_VALUES_MAX], Pd_phi_[JGX_N_VALUES_MAX];
    const jgx::view<Real, 1> Pd(Pd_, pe), Pd_s(Pd_s_, pe);
    const jgx::view<Real, 1> Pd_t(Pd_t_, pe), Pd_phi(Pd_phi_, pe);

    interp::interp_PRZP_1(element_list, node_list, ie, i_v, n_v, s, t, phi,
                          true, Pd, Pd_s, Pd_t, Pd_phi,
                          R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi);

    double dt;
    if (fabs(time_now - time_prev) > 1e-10 && !is_static) {
      dt = 1.0/(time_now - time_prev);
      const double df = (time_now - time)*dt;
      for (int i = 0; i < n_v; ++i) {
        P    (i) -= df*Pd    (i);
        P_s  (i) -= df*Pd_s  (i);
        P_t  (i) -= df*Pd_t  (i);
        P_phi(i) -= df*Pd_phi(i);
      }
    } else {
      dt = 1.0/t_jorek;
    }

    for (int i = 0; i < n_v; ++i) P_time(i) = Pd(i)*dt;
  } // interp_PRZP_1
};

using fields_interp_linear_set_aos = fields_interp_linear_set<layout_stride>;
using fields_interp_linear_set_soa = fields_interp_linear_set<layout_left>;

template <class L, class Real = double, class Int = int>
JGX_HD inline fields_interp_linear_set<L, Real, Int>
make_fields_interp_linear_set(const element_set<L, Real, Int>& el,
                              const node_set<L, Real, Int>& nd,
                              const void* interp_base, const jgx_record_desc& r) {
  fields_interp_linear_set<L, Real, Int> f;
  fill_fields_base(f, el, nd, interp_base, r);
  f.time_now  = jgx::record_scalar<Real>(interp_base, r.field[JGX_FIL_TIME_NOW ]);
  f.time_prev = jgx::record_scalar<Real>(interp_base, r.field[JGX_FIL_TIME_PREV]);
  return f;
}

/* Address an existing Fortran jorek_fields_interp_linear in place. All three
 * bases -- the two meshes and the interpolator -- come from inside the caller's
 * select type, and all three get their layout from the registry. The
 * interpolator is one record, so its count is implicit. */
inline fields_interp_linear_set_aos
fields_interp_linear_set_from_registry(void* el_base, std::size_t n_elements,
                                       void* nd_base, std::size_t n_nodes,
                                       const void* interp_base) {
  return make_fields_interp_linear_set<layout_stride, double, int>(
      element_set_from_registry(el_base, n_elements),
      node_set_from_registry(nd_base, n_nodes),
      interp_base,
      jgx::registered_record(JGX_REC_FIELDS_INTERP_LINEAR, JGX_FIL_COUNT));
}

} /* namespace jorek */

#endif //FIELDS_LINEAR_H
