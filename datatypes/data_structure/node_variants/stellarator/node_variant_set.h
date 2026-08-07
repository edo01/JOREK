/* node_variants/stellarator/node_variant_set.h -- the node set of a stellarator model.
 *
 * It defines the node_model_set_* and node_model_set_from_registry.
 */
#ifndef JOREK_NODE_VARIANT_SET_H
#define JOREK_NODE_VARIANT_SET_H

#include <cstddef>
#include "datatypes/data_structure/node_variants/stellarator/node_stellarator_set.h"

namespace jorek {

template <class L, class Real = double, class Int = int>
using node_model_set = node_stellarator_set<L, Real, Int>;

using node_model_set_aos = node_model_set<layout_stride>;
using node_model_set_soa = node_model_set<layout_left>;

inline node_model_set_aos node_model_set_from_registry(void* base,
                                                       std::size_t n_nodes) {
  return node_stellarator_set_from_registry(base, n_nodes);
}

} /* namespace jorek */

#endif /* JOREK_NODE_VARIANT_SET_H */
