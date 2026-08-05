/* jgx/jorek/jorek_settings.h -- the array extents of models/mod_settings.f90.
 *
 * The build reads them out of that file and passes them as -D, so a
 * reconfigured JOREK cannot leave the C++ side sized for the old parameters.
 * See cmake/JorekModelConfig.cmake and defaults.mk.
 */
#ifndef JOREK_SETTINGS_H
#define JOREK_SETTINGS_H

#if !defined(JGX_N_TOR) || !defined(JGX_N_COORD_TOR) || !defined(JGX_N_ORDER) \
    || !defined(JGX_N_DEGREES) || !defined(JGX_N_VERTEX_MAX) || !defined(JGX_N_VALUES_MAX)
#error "mod_settings.f90 extents missing; compile through the CMake or Makefile build"
#endif

#endif /* JOREK_SETTINGS_H */
