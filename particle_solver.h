/*!
  \file particle_solver.h
  \brief Solver routines for particle position and velocity (header file)
  \author Y. Nakayama
  \date 2006/10/16
  \version 1.2
  \todo documentation
 */
#ifndef PARTICLE_SOLVER_H
#define PARTICLE_SOLVER_H

#include <math.h>

#include "ewald_wrapper.h"
#include "input.h"
#include "md_force.h"
#include "particle_rotation_solver.h"
#include "periodic_boundary.h"
#include "rigid.h"
#include "rigid_body.h"
#include "variable.h"
#include "wall.h"

enum ITER { start_iter, new_iter, reset_iter, end_iter };

/*!
  \brief Update particle positions using the Euler
  forward method
  \details \f{align*}{
  \vec{R}_i^{n+1} &= \vec{R}_i^{n} + h \vec{V}_i^{n}
  \f}
  \param[in,out] p particle data
  \param[in] jikan time data
 */
void MD_solver_position_Euler(Particle *p, const CTime &jikan);

/*!
  \brief Update particle positions using a
  second-order Adams-Bashforth scheme
  \details \f{align*}{
  \vec{R}_i^{n+1} &= \vec{R}_i^{n} + \frac{h}{2}\left(3\vec{V}_i^{n} -
  \vec{V}_i^{n-1}\right)
    \f}
    \param[in,out] p particle data
  \param[in] jikan time data
 */
void MD_solver_position_AB2(Particle *p, const CTime &jikan);

/*!
  \brief  Update particle velocities using Euler forward method
  \details Saves data for old velocities and forces
  \param[in,out] p particle data
  \param[in] jikan time data
 */
void MD_solver_velocity_Euler(Particle *p, const CTime &jikan);
/*!
  \brief Update particle velocities using a second-order
  Adams-Bashforth scheme
  \details Saves data for old velocities and forces
  \param[in,out] p particle data
  \param[in] jikan time data
 */
void MD_solver_velocity_AB2_hydro(Particle *p, const CTime &jikan);

/*!
  \brief Update particle velocities for swimming particles
  \details Automatically chooses appropriate Euler/Adams-Bashforth
  scheme depending on the jikan step. Part of iterative solution for
  particle velocities. Only slip force changes, other quantities
  should only be computed at the first iteration.
 */
void MD_solver_velocity_slip_iter(Particle *p, const CTime &jikan, const ITER &iter_flag);

// Oblique coordinates
/*!
  \brief Update particle positions and orientations using the Euler
  forward method for sheared systems with Lees-Edwards PBC
 */
void MD_solver_position_Euler_OBL(Particle *p, const CTime &jikan);
/*!
  \brief Update particle positions and orientations using a
  second-order Adams-Bashforth scheme for sheared systems with Lees-Edwards PBC
 */
void MD_solver_position_AB2_OBL(Particle *p, const CTime &jikan);
void MD_solver_velocity_Euler_OBL(Particle *p, const CTime &jikan);
void MD_solver_velocity_AB2_hydro_OBL(Particle *p, const CTime &jikan);

inline void Force(Particle *p) {
    if (LJ_truncate >= 0) {
        Calc_f_Lennard_Jones(p);
    }

    if (G != 0.0) {
        Add_f_gravity(p);
    }
    if (SW_PT == chain) {
        Calc_anharmonic_force_chain(p, Distance0);
    }

    if (SW_WALL != NO_WALL) {
        Add_f_wall(p);
    }

    if (SW_QUINCKE == QUINCKE_ON) {
        Calc_harmonic_torque_quincke(p);
    }

    if (SW_MULTIPOLE == MULTIPOLE_ON) {
        if (ewald_param.m_image == true) {
            Calc_multipole_interaction_force_torque_with_image(p);
        } else {
            Calc_multipole_interaction_force_torque(p);
        }
        
    }
}

inline void Force_OBL(Particle *p) {
    dev_shear_stress_lj = dev_shear_stress_rot = 0.0;
    rigid_dev_shear_stress_lj = rigid_dev_shear_stress_rot = 0.0;

    if (LJ_truncate >= 0) Calc_f_Lennard_Jones_OBL(p);

    if (G != 0.0) Add_f_gravity(p);

    if (SW_PT == chain) Calc_anharmonic_force_chain(p, Distance0_OBL);

    dev_shear_stress_lj *= Ivolume;
    dev_shear_stress_rot *= Ivolume;

    rigid_dev_shear_stress_lj *= Ivolume;
    rigid_dev_shear_stress_rot *= Ivolume;
}

inline void Pinning(Particle *p) {
    if (SW_PT != rigid) {
#pragma omp parallel for
        for (int i = 0; i < N_PIN; i++) {
            for (int d = 0; d < DIM; d++) {
                p[Pinning_Numbers[i]].v[d] = 0.0;
            }
        }
        for (int i = 0; i < N_PIN_ROT; i++) {
            for (int d = 0; d < DIM; d++) {
                p[Pinning_ROT_Numbers[i]].omega[d] = 0.0;
            }
        }
    }
}
#endif
