/* jgx/layout_policy.h -- Compile-time layout policy aliases. 
 * Kernel bodies index these arrays only through views
 * typed on these aliases, so swapping an axis is a recompile, never a kernel
 * edit.
 * 
 * The change of view is always done C/C++ side, leaving untouched the 
 * fortran data structures.
 */

 /**
  * @todo:  node/elem: these are READ arrays whose device buffer is a byte copy of the
  * column-major Fortran array, so switching their view strides without a
  * matching repack-on-push would misread. 
  * 
  * @todo: how does this scale with the number of data structures ported? We will end up
  * with a big list of data structures all concentrated here. In addition to that, we cannot 
  * have one define for each buffer. What could we use? A script like utils/config.sh? Maybe
  * inside CMake there is already something
  * 
  */

#ifndef JGX_LAYOUT_POLICY_H
#define JGX_LAYOUT_POLICY_H

#include "jgx/view.h"

/*
 * These are left as an example
 */
#ifndef JGX_NODE_LAYOUT_FAST
#  define JGX_NODE_LAYOUT_FAST 0
#endif
#ifndef JGX_ELEM_LAYOUT_FAST
#  define JGX_ELEM_LAYOUT_FAST 0
#endif
#ifndef JGX_FEEDBACK_LAYOUT_FAST
#  define JGX_FEEDBACK_LAYOUT_FAST 1
#endif

/* Full axis-order control (Q4). Define JGX_FEEDBACK_PERM to a fastest-first
 * axis list to pick ANY of the 3! orders for feedback(nelem,nvar,4), e.g.
 *   -DJGX_FEEDBACK_PERM=1,0,2
 * When unset, fall back to the fast/slow pair (perm<0,1,2> / perm<2,1,0>). */
#ifndef JGX_FEEDBACK_PERM
#  if JGX_FEEDBACK_LAYOUT_FAST
#    define JGX_FEEDBACK_PERM 0,1,2     /* element index fastest */
#  else
#    define JGX_FEEDBACK_PERM 2,1,0     /* element index slowest */
#  endif
#endif

namespace jgx {

/* Feedback: switchable, sum-invariant -- any axis permutation is legal. */
using feedback_layout = layout_perm<JGX_FEEDBACK_PERM>;

} /* namespace jgx */

#endif /* JGX_LAYOUT_POLICY_H */
