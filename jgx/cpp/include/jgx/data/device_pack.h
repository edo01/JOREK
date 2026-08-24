/* jgx/data/device_pack.h -- packed copy of a record array, in device memory.
 *
 * One SoA packed block plus one staging buffer for the AoS block, allocated from the
 * backend and freed with the object. Its base is what a set's from_soa takes,
 * exactly as a record array's base is what from_aos takes:
 *
 *     jgx::data::device_pack pk(
 *         jgx::data::registered_record(JGX_REC_ELEMENT, JGX_EF_COUNT),
 *         n_elements);
 *     pk.upload(el_base);
 *     auto el = jorek::element_set_soa::from_soa(pk.base(), pk.record(),
 *                                                n_elements);
 *
 * The transfers involve no host code -- upload() is one contiguous H2D of the
 * whole AoS block followed by a transpose kernel. The host alternative would
 * walk the array one component element at a time, an AoS component not being
 * contiguous across records.
 *
 */
#ifndef JGX_DATA_DEVICE_PACK_H
#define JGX_DATA_DEVICE_PACK_H

#include <cstddef>

#include "jgx/jgx_c_api.h"
#include "jgx/data/record_api.h"
#include "jgx/data/pack.h"

namespace jgx::data {

class device_pack {
 public:
  /* `r` is held by reference: registrations are static and outlive any pack.
   * Two allocations whatever the record is -- the staging block and the packed
   * one, the latter carved by jgx::data::soa_field_offset, whose padding gives
   * each field the alignment it used to get from an allocation of its own. */
  device_pack(const jgx_record_desc& r, std::size_t n_records)
      : r_(r), n_records_(n_records),
        aos_bytes_(n_records * r.record_stride_bytes),
        soa_bytes_(jgx::data::soa_block_bytes(r, n_records)) {
    aos_ = jgx_c_alloc(aos_bytes_);
    soa_ = jgx_c_alloc(soa_bytes_);
  }

  ~device_pack() {
    jgx_c_free(soa_);
    jgx_c_free(aos_);
  }

  device_pack(const device_pack&) = delete;
  device_pack& operator=(const device_pack&) = delete;

  /* Push + pack an AoS data structure from the host. */
  void upload(const void* aos_base) const {
    jgx_c_push(aos_, aos_base, aos_bytes_);
    jgx_c_pack(soa_, &r_, aos_, n_records_);
  }

  /* Unpack + pull a SoA data structure from the device. */
  void download(void* aos_base) const { download(aos_base, soa_); }

  /* Unpack + pull a SoA data structure from the device by passing
    the source when this is not the owned one. */
  void download(void* aos_base, const void* soa_base) const {
    jgx_c_unpack(aos_, soa_base, &r_, n_records_);
    jgx_c_pull(aos_base, aos_, aos_bytes_);
  }

  /* Return the SoA packed block. */
  void* base() const noexcept { return soa_; }
  const jgx_record_desc& record() const noexcept { return r_; }

  /* One field's buffer inside the block -- for a transfer of a single
   * component. */
  void* field(int field_id) const noexcept {
    return jgx::data::soa_field_ptr(soa_, r_, field_id, n_records_);
  }

  std::size_t field_bytes(int field_id) const noexcept {
    return jgx::data::field_bytes(r_.field[field_id], n_records_);
  }

  std::size_t n_records() const noexcept { return n_records_; }

  /* Everything this object holds on the device, staging buffer and block padding
   * included: the per-call transfer volume. */
  std::size_t bytes() const noexcept { return aos_bytes_ + soa_bytes_; }

 private:
  const jgx_record_desc& r_;
  std::size_t n_records_;
  std::size_t aos_bytes_;
  std::size_t soa_bytes_;
  void* aos_ = nullptr; // device pointer to the aos buffer
  void* soa_ = nullptr; // device pointer to the soa memory
};

} /* namespace jgx::data */

#endif /* JGX_DATA_DEVICE_PACK_H */
