#include "grids/mod_neighbours/neighbours.h"
#include "datatypes/data_structure/element_set.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    /* mod_neighbours::coord_in_neighbour.
     *
     * node_list is not forwarded: the Fortran takes it but never reads it.
     * i_to and bad_i_to are the two outputs the facade acts on -- see
     * neighbours.h for what i_to's sentinels mean and why bad_i_to exists.
     */
    void jgx_host_coord_in_neighbour(void* el_base, const int32_t n_elements,
                                     const int32_t i_from0,
                                     int32_t* i_to, double* st, int32_t* bad_i_to) {

        const auto el = jorek::element_set_aos::from_registry(el_base, static_cast<std::size_t>(n_elements));

        int to = 0, bad = 0;
        neighbours::coord_in_neighbour(el, static_cast<std::size_t>(i_from0), to, st, bad);

        *i_to     = static_cast<int32_t>(to);
        *bad_i_to = static_cast<int32_t>(bad);
    }
}
