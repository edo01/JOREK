/*
Setting parameters for the JOREK code.
*/

#define n_tor               7           /* number of toroidal harmonics */
#define n_coord_tor         1           /* number of toroidal harmonics in (R, Z) coordinates */
#define l_pol_domm          0           /* highest poloidal mode in the Dommaschk potentials */
#define n_period            1           /* periodicity in toroidal direction */
#define n_coord_period      1           /* periodicity of the device in toroidal direction: equivalent to number of field periods */
#define n_plane             12          /* number of toroidal angles */
#define n_order             3           /* polynomial order of the finite element basis */
#define n_nodes_max         90001       /* maximum number of nodes */
#define n_elements_max      90001       /* maximum number of elements */
#define n_boundary_max      9001        /* maximum number of boundary elements */
#define n_pieces_max        9001        /* maximum number of pieces */


/*  The following line is needed by ./util/config.sh:
    #SETTINGS# n_tor n_coord_tor l_pol_domm n_period n_coord_period n_plane n_order n_nodes_max n_elements_max n_boundary_max n_pieces_max 
*/
    
/* --- a few constants that should not be touched */
#define n_dim               2                               /* number of dimensions */
#define n_vertex_max        4                               /* maximum number of corners per element */
#define n_degrees_1d       ((n_order+1)/2)                 /* degrees of freedom per variable per node in 1D (used for boundary conditions) */
#define n_degrees          (n_degrees_1d*n_degrees_1d)      /* degrees of freedom per variable per node in 2D */
#define nref_max           10
#define n_ref_list         10
#define n_var_max           8