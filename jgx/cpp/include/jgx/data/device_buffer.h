/* jgx/data/device_buffer.h -- a scratch allocation on the device.
 *
 * The counterpart of jgx::data::device_pack for memory that is not a record
 * array: histograms, staging rows, a feedback buffer, a struct of diagnostics.
 * The pack owns a record array and knows how to transpose it; this owns bytes.
 */
#ifndef JGX_DATA_DEVICE_BUFFER_H
#define JGX_DATA_DEVICE_BUFFER_H

#include <cstddef>

#include "jgx/jgx_c_api.h"

namespace jgx {

class device_buffer {
 public:
  explicit device_buffer(std::size_t n_bytes)
      : n_bytes_(n_bytes), p_(jgx_c_alloc(n_bytes)) {
    jgx_c_memset(p_, 0, n_bytes_);
  }
  ~device_buffer() { jgx_c_free(p_); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  void* get() const noexcept { return p_; }
  template <class T> T* as() const noexcept { return static_cast<T*>(p_); }
  std::size_t bytes() const noexcept { return n_bytes_; }

 private:
  std::size_t n_bytes_;
  void* p_;
};

/* Blocks needed to cover n items at the given block size. */
inline unsigned grid_for(std::size_t n, unsigned block) {
  return static_cast<unsigned>((n + block - 1) / block);
}

} /* namespace jgx */

#endif /* JGX_DATA_DEVICE_BUFFER_H */
