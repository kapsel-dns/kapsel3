/*!
  \file sp_3d_ns.cxx
  \author Y. Nakayama
  \date 2006/11/14
  \version 1.5
  \brief Main program file
 */
#include "sp_3d_ns.h"

void (*Time_evolution)(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan);
void (*Time_evolution_fdm)(double **&u, double *Pressure, double **f, Particle *p, CTime &jikan);

void Time_evolution_noparticle(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan) {
    const Index_range *ijk_range   = ijk_range_two_third_filter;
    const int          n_ijk_range = n_ijk_range_two_third_filter;
    if (SW_EQ == Navier_Stokes) {
        NS_solver_slavedEuler(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p);
    } else if (SW_EQ == Shear_Navier_Stokes) {
        NS_solver_slavedEuler_Shear_PBC(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p, Shear_force);
    } else if (SW_EQ == Electrolyte) {
        NSsolute_solver_Euler(zeta, jikan, uk_dc, Concentration, p, ijk_range, n_ijk_range);
    }
}

void Time_evolution_noparticle_fdm(double **u, double *Pressure, Particle *p, CTime &jikan) {
    if (SW_NSST == explicit_scheme) {
        NS_solver_slavedEuler_explicit(u, Pressure, p, jikan);
    } else if (SW_NSST == implicit_scheme) {
        NS_solver_slavedEuler_implicit(u, Pressure, p, jikan);
    }
}

void Time_evolution_noparticle_OBL_fdm(double **&u, double *Pressure, Particle *p, CTime &jikan) {
    if (SW_NSST == explicit_scheme) {
        NS_solver_slavedEuler_Shear_OBL_explicit(u, Pressure, p, jikan);
    } else if (SW_NSST == implicit_scheme) {
        NS_solver_slavedEuler_Shear_OBL_implicit(u, Pressure, p, jikan);
    }
}

void Time_evolution_hydro(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan) {
    // Update of Fluid Velocity Field
    Time_evolution_noparticle(zeta, uk_dc, f, p, jikan);

    if (Particle_Number >= 0) {
        if (FIX_CELL) {  // time-dependent average pressure gradient
            for (int d = 0; d < DIM; d++) {
                if (FIX_CELLxyz[d]) {
                    uk_dc[d] = 0.0;
                }
            }
        }

        for (int d = 0; d < (DIM - 1); d++) {
            Truncate_two_third_rule(zeta[d]);
        }

        Zeta_k2u(zeta, uk_dc, u);

        if (!Fixed_particle) {
            if (jikan.ts == 0) {
                MD_solver_position_Euler(p, jikan);
            } else {
                MD_solver_position_AB2(p, jikan);
            }
        }
        Reset_phi(phi);
        if (SW_WALL != NO_WALL) {
            Copy_v1(phi_sum, phi_wall);
        } else {
            Reset_phi(phi_sum);
        }
        Make_phi_particle_sum(phi, phi_sum, p);

        if (SW_EQ == Electrolyte) {
            double *rescale_factor = new double[N_spec];
            Rescale_solute(rescale_factor, Total_solute, Concentration, p, phi, up[0]);
            delete[] rescale_factor;
        }

        if (SW_EQ == Electrolyte) {
            {
                Reset_u(up);
                Calc_f_hydro_correct_precision(p, phi_sum, u, jikan);
                for (int n = 0; n < Particle_Number; n++) {
                    for (int d = 0; d < DIM; d++) {
                        p[n].f_hydro1[d]      = p[n].f_hydro[d];
                        p[n].torque_hydro1[d] = p[n].torque_hydro[d];
                    }
                }
            }
            Make_Coulomb_force_x_on_fluid(f, p, Concentration, up[0], up[1], jikan);

            double dmy = jikan.dt_fluid * IRHO;
#pragma omp parallel for
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                        u[0][(i * NY * NZ_) + (j * NZ_) + k] += (f[0][(i * NY * NZ_) + (j * NZ_) + k] * dmy);
                        u[1][(i * NY * NZ_) + (j * NZ_) + k] += (f[1][(i * NY * NZ_) + (j * NZ_) + k] * dmy);
                        u[2][(i * NY * NZ_) + (j * NZ_) + k] += (f[2][(i * NY * NZ_) + (j * NZ_) + k] * dmy);
                    }
                }
            }
            Solenoidal_u(u);
        }

        {  // Calculation of hydrodynamic force

            Reset_u(up);
            Calc_f_hydro_correct_precision(p, phi_sum, u, jikan);  // hydrodynamic force

            if (!SW_JANUS_SLIP) {
                if (!Fixed_particle) {
                    if (jikan.ts == 0) {
                        MD_solver_velocity_Euler(p, jikan);
                    } else {
                        MD_solver_velocity_AB2_hydro(p, jikan);
                    }
                }
            } else {  // Self-Consistent slip force

                int slip_converge = 0;
                int slip_iter     = 0;
                Make_particle_momentum_factor(u, p);
                Update_slip_particle_velocity(p, slip_iter);  // initial particle velocity for slip profile
                while (!slip_converge) {
                    Reset_u(up);

                    Make_force_u_slip_particle(up, u, p, jikan);
                    Solenoidal_u(up);
                    Calc_f_slip_correct_precision(p, up, jikan);  // slip force
                    Add_f_particle(up, u);                        // up += u

                    // Update particle velocity
                    if (!Fixed_particle) {
                        if (slip_iter == 0) {
                            MD_solver_velocity_slip_iter(p, jikan, start_iter);
                        } else {
                            MD_solver_velocity_slip_iter(p, jikan, new_iter);
                        }
                    }
                    if (PINNING) {
                        Pinning(p);
                    }
                    slip_iter++;

                    if (Slip_particle_convergence(p) < MAX_SLIP_TOL || slip_iter == MAX_SLIP_ITER) {
                        slip_converge = 1;
                        MD_solver_velocity_slip_iter(p, jikan, end_iter);
                    } else {
                        Update_slip_particle_velocity(p, slip_iter);  // use new particle velocity for new slip profile
                        MD_solver_velocity_slip_iter(p, jikan, reset_iter);
                    }
                }  // slip_convergence
                if (slip_iter == MAX_SLIP_ITER) {
                    fprintf(stderr, "#Warning: increase MAX_SLIP_ITER (%d)\n", jikan.ts);
                }

                Swap_mem(u, up);
            }  // slip
        }

        if (kBT > 0. && SW_EQ != Electrolyte) {
            Add_random_force_thermostat(p, jikan);
        }

        if (PINNING) {
            Pinning(p);
        }

        {
            // Reset_phi_u(phi, up);
            // Make_phi_u_particle(phi, up, p);
            Reset_u(up);
            Make_u_particle_sum(up, phi_sum, p);
            Make_f_particle_dt_sole(f, u, up, phi);
            Add_f_particle(u, f);
        }

        if (Shear_AC) {
#pragma omp parallel for
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                        ucp[0][(i * NY * NZ_) + (j * NZ_) + k] = u[0][(i * NY * NZ_) + (j * NZ_) + k];
                    }
                }
            }
        }

        U2zeta_k(zeta, uk_dc, u);
    }
}

void Time_evolution_hydro_fdm(double **&u, double *Pressure, double **f, Particle *p, CTime &jikan) {
    // Update of Fluid Velocity Field
    Time_evolution_noparticle_fdm(u, Pressure, p, jikan);

    if (Particle_Number >= 0) {
        if (FIX_CELL) {  // time-dependent average pressure gradient
            for (int d = 0; d < DIM; d++) {
                if (FIX_CELLxyz[d]) {
                    dc_offset(u[d]);
                }
            }
        }

        if (!Fixed_particle) {
            if (jikan.ts == 0) {
                MD_solver_position_Euler(p, jikan);
            } else {
                MD_solver_position_AB2(p, jikan);
            }
        }
        Reset_phi(phi);
        if (SW_WALL != NO_WALL) {
            Copy_v1(phi_sum, phi_wall);
        } else {
            Reset_phi(phi_sum);
        }
        Make_phi_particle_sum(phi, phi_sum, p);
        Make_phi_p(phi_p,phi,phi_wall);
        // Calculation of hydrodynamic force

        Reset_u(up);
        Calc_f_hydro_correct_precision(p, phi_sum, u, jikan);  // hydrodynamic force

        if (!SW_JANUS_SLIP) {
            if (!Fixed_particle) {
                if (jikan.ts == 0) {
                    MD_solver_velocity_Euler(p, jikan);
                } else {
                    MD_solver_velocity_AB2_hydro(p, jikan);
                }
            }
        } else {  // Self-Consistent slip force

            int slip_converge = 0;
            int slip_iter     = 0;
            Make_particle_momentum_factor(u, p);
            Update_slip_particle_velocity(p, slip_iter);  // initial particle velocity for slip profile
            while (!slip_converge) {
                Reset_u(up);

                Make_force_u_slip_particle(up, u, p, jikan);
                Solenoidal_u(up);
                Calc_f_slip_correct_precision(p, up, jikan);  // slip force
                Add_f_particle(up, u);                        // up += u

                // Update particle velocity
                if (!Fixed_particle) {
                    if (slip_iter == 0) {
                        MD_solver_velocity_slip_iter(p, jikan, start_iter);
                    } else {
                        MD_solver_velocity_slip_iter(p, jikan, new_iter);
                    }
                }
                if (PINNING) {
                    Pinning(p);
                }
                slip_iter++;

                if (Slip_particle_convergence(p) < MAX_SLIP_TOL || slip_iter == MAX_SLIP_ITER) {
                    slip_converge = 1;
                    MD_solver_velocity_slip_iter(p, jikan, end_iter);
                } else {
                    Update_slip_particle_velocity(p, slip_iter);  // use new particle velocity for new slip profile
                    MD_solver_velocity_slip_iter(p, jikan, reset_iter);
                }
            }  // slip_convergence
            if (slip_iter == MAX_SLIP_ITER) {
                fprintf(stderr, "#Warning: increase MAX_SLIP_ITER (%d)\n", jikan.ts);
            }

            Swap_mem(u, up);
        }  // slip

        if (kBT > 0.) {
            Add_random_force_thermostat(p, jikan);
        }

        if (PINNING) {
            Pinning(p);
        }

        Reset_u(up);
        Make_u_particle_sum(up, phi_sum, p);

        Make_f_particle_dt_nonsole(f, u, up, phi);
        Add_f_particle(u, f);

        //----------------------------------------
        Solenoidal_u(u);
        //----------------------------------------

        if (Shear_AC) {
#pragma omp parallel for
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                        ucp[0][(i * NY * NZ_) + (j * NZ_) + k] = u[0][(i * NY * NZ_) + (j * NZ_) + k];
                    }
                }
            }
        }
    }

    if (PHASE_SEPARATION) {
        if (SW_CHST == explicit_scheme) {
            Cpy_v1(psi_o, psi);
            if (SW_WALL != NO_WALL) {
                Calc_cp_wall(phi, phi_p, phi_wall, psi, cp);
            } else {
                Calc_cp(phi, psi, cp);
            }
            Update_psi_euler(psi, u, phi, cp, jikan);
        } else if (SW_CHST == implicit_scheme) {
            if (jikan.ts < 2) {
#ifdef _LIS_SOLVER
                CH_solver_implicit_euler(psi, psi_o, phi, u, jikan, is_ch, ie_ch);
#else
                CH_solver_implicit_euler(psi, psi_o, phi, u, jikan, 0, NX * NY * NZ);
#endif
            } else {
#ifdef _LIS_SOLVER
                CH_solver_implicit_bdfab(psi, psi_o, phi, u, jikan, is_ch, ie_ch);
#else
                CH_solver_implicit_bdfab(psi, psi_o, phi, u, jikan, 0, NX * NY * NZ);
#endif
            }
        }
    }
}

void Time_evolution_hydro_OBL(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan) {
    // AC
    Angular_Frequency = PI2 * Shear_frequency;
    //    double ifreq = 1/Angular_Frequency;
    //    double shear_amp = Shear_rate * ifreq;
    //
    const Index_range *ijk_range   = ijk_range_two_third_filter;
    const int          n_ijk_range = n_ijk_range_two_third_filter;
    NS_solver_slavedEuler_Shear_OBL(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p, Shear_force);

    if (Particle_Number >= 0) {
        if (FIX_CELL) {  // time-dependent average pressure gradient
            for (int d = 0; d < DIM; d++) {
                if (FIX_CELLxyz[d]) {
                    uk_dc[d] = 0.0;
                }
            }
        }
        /*
        for(int d = 0; d < DIM - 1; d++){
                Truncate_two_third_rule(zeta[d]);
        }
        */

        Zeta_k2u_k_OBL(zeta, uk_dc, u);
        U_k2u(u);
        if (!Shear_AC) {
            Shear_rate_eff = Shear_rate;
        } else {
            Shear_rate_eff = Shear_rate * cos(Angular_Frequency * jikan.dt_fluid * jikan.ts);
        }
        degree_oblique += Shear_rate_eff * jikan.dt_fluid;
        if (degree_oblique >= XYaspect) {
            int flag = 0;
            Reset_U_OBL(ucp, u, flag);
            Swap_mem(u, ucp);
            degree_oblique -= XYaspect;
        } else if (degree_oblique <= -XYaspect) {
            int flag = 1;
            Reset_U_OBL(ucp, u, flag);
            Swap_mem(u, ucp);
            degree_oblique += XYaspect;
        }
        Update_K2_OBL();

        Copy_v3(ucp, u);
        Transform_obl_u(ucp, oblique2cartesian);

        // u   -> velocity field in oblique coordinates
        // ucp -> velocity fild in cartesian coordinates
        // Calc_shear_rate_eff();
        // AC
        sreff_old = Shear_rate_eff;
        if (!Shear_AC) Calc_shear_rate_eff();
        //
        // End Deformation

        if (!Fixed_particle) {
            if (jikan.ts == 0) {
                MD_solver_position_Euler_OBL(p, jikan);
            } else {
                MD_solver_position_AB2_OBL(p, jikan);
            }
        }

        {  // Calculation of hydrodynamic force
            Reset_phi(phi);
            Reset_phi(phi_sum);
            Reset_u(up);

            Make_phi_particle_sum_OBL(phi, phi_sum, p);
            Calc_f_hydro_correct_precision_OBL(p, phi_sum, ucp, jikan);
            Calc_Reynolds_shear_stress(ucp, Inertia_stress);
        }

        if (!Fixed_particle) {  // Update of Particle Velocity
            if (jikan.ts == 0) {
                MD_solver_velocity_Euler_OBL(p, jikan);
            } else {
                MD_solver_velocity_AB2_hydro_OBL(p, jikan);
            }
        }

        if (kBT > 0. && SW_EQ != Electrolyte) {
            Add_random_force_thermostat(p, jikan);
        }

        if (PINNING) {
            Pinning(p);
        }

        {
            Reset_u(up);
            Make_u_particle_sum_OBL(up, phi_sum, p);

            Make_f_particle_dt_nonsole(f, ucp, up, phi);
            Transform_obl_u(f, cartesian2oblique);
            Add_f_particle(u, f);
        }

        U2u_k(u);
        Solenoidal_uk_OBL(u);

        U_k2zeta_k_OBL(u, zeta, uk_dc);
    }
}

void Time_evolution_hydro_OBL_fdm(double **&u, double *Pressure, double **f, Particle *p, CTime &jikan) {
    // AC
    Angular_Frequency = PI2 * Shear_frequency;

    const Index_range *ijk_range   = ijk_range_two_third_filter;
    const int          n_ijk_range = n_ijk_range_two_third_filter;

    Time_evolution_noparticle_OBL_fdm(u, Pressure, p, jikan);

    if (Particle_Number >= 0) {
        if (FIX_CELL) {  // time-dependent average pressure gradient
            for (int d = 0; d < DIM; d++) {
                if (FIX_CELLxyz[d]) {
                    dc_offset(u[d]);
                }
            }
        }
        if (!Shear_AC) {
            Shear_rate_eff = Shear_rate;
        } else {
            Shear_rate_eff = Shear_rate * cos(Angular_Frequency * jikan.dt_fluid * jikan.ts);
        }
        degree_oblique += Shear_rate_eff * jikan.dt_fluid;
        if (degree_oblique >= XYaspect) {
            int flag = 0;
            Reset_U_OBL(ucp, u, flag);
            Swap_mem(u, ucp);

            Reset_U_OBL(u_o_cpy, u_o, flag);
            Swap_mem(u_o, u_o_cpy);

            if (PHASE_SEPARATION) {
                Reset_A_OBL(psicp, psi, flag);
                Swap_mem(psi, psicp);

                Reset_A_OBL(psicp_o, psi_o, flag);
                Swap_mem(psi_o, psicp_o);
            }

            degree_oblique -= XYaspect;
        } else if (degree_oblique <= -XYaspect) {
            int flag = 1;
            Reset_U_OBL(ucp, u, flag);
            Swap_mem(u, ucp);

            Reset_U_OBL(u_o_cpy, u_o, flag);
            Swap_mem(u_o, u_o_cpy);

            if (PHASE_SEPARATION) {
                Reset_A_OBL(psicp, psi, flag);
                Swap_mem(psi, psicp);

                Reset_A_OBL(psicp_o, psi_o, flag);
                Swap_mem(psi_o, psicp_o);
            }

            degree_oblique += XYaspect;
        }
        Update_K2_OBL();

        Copy_v3(ucp, u);
        Transform_obl_u(ucp, oblique2cartesian);

        if (PHASE_SEPARATION) {
            Copy_v1(psicp, psi);
            Transform_obl_a(psicp, oblique2cartesian);
        }

        // u   -> velocity field in oblique coordinates
        // ucp -> velocity field in cartesian coordinates
        // AC
        if (!Shear_AC) Calc_shear_rate_eff();
        //
        // End Deformation

        if (!Fixed_particle) {
            if (jikan.ts == 0) {
                MD_solver_position_Euler_OBL(p, jikan);
            } else {
                MD_solver_position_AB2_OBL(p, jikan);
            }
        }

        {  // Calculation of hydrodynamic force
            Reset_phi(phi);
            Reset_phi(phi_sum);
            Reset_u(up);

            Make_phi_particle_sum_OBL(phi, phi_sum, p);
            Calc_f_hydro_correct_precision_OBL(p, phi_sum, ucp, jikan);
            Calc_Reynolds_shear_stress(ucp, Inertia_stress);
        }

        if (!Fixed_particle) {  // Update of Particle Velocity
            if (jikan.ts == 0) {
                MD_solver_velocity_Euler_OBL(p, jikan);
            } else {
                MD_solver_velocity_AB2_hydro_OBL(p, jikan);
            }
        }

        if (kBT > 0. && SW_EQ != Electrolyte) {
            Add_random_force_thermostat(p, jikan);
        }

        if (PINNING) {
            Pinning(p);
        }

        {
            Reset_u(up);
            Make_u_particle_sum_OBL(up, phi_sum, p);

            Make_f_particle_dt_nonsole(f, ucp, up, phi);
            Transform_obl_u(f, cartesian2oblique);
            Add_f_particle(u, f);
        }

        if (PHASE_SEPARATION) {
            Copy_v1(phi_obl, phi);
            A2a_oblique(phi_obl);
        }

        //--------------------------------
        U2u_k(u);
        Solenoidal_uk_OBL(u);
        U_k2u(u);
        //--------------------------------
    }

    if (PHASE_SEPARATION) {
        if (SW_CHST == explicit_scheme) {
            Cpy_v1(psi_o, psi);
            Calc_cp_OBL(phi_obl, psi, cp, degree_oblique);
            Update_psi_euler_OBL(psi, u, cp, jikan, degree_oblique);
        } else if (SW_CHST == implicit_scheme) {
            if (jikan.ts < 2) {
#ifdef _LIS_SOLVER
                CH_solver_implicit_euler_OBL(psi, psi_o, phi_obl, u, jikan, degree_oblique, is_ch, ie_ch);
#else
                CH_solver_implicit_euler_OBL(psi, psi_o, phi_obl, u, jikan, degree_oblique, 0, NX * NY * NZ);
#endif
            } else {
#ifdef _LIS_SOLVER
                CH_solver_implicit_bdfab_OBL(psi, psi_o, phi_obl, u, jikan, degree_oblique, is_ch, ie_ch);
#else
                CH_solver_implicit_bdfab_OBL(psi, psi_o, phi_obl, u, jikan, degree_oblique, 0, NX * NY * NZ);
#endif
            }
        }
    }
}

inline void Mem_alloc_var(double **zeta) {
    Mem_alloc_NS_solver();
    if (SW_EQ == Navier_Stokes || SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
        SW_EQ == Navier_Stokes_FDM || SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM ||
        SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
        ucp = (double **)malloc(sizeof(double *) * DIM);
        for (int d = 0; d < DIM; d++) {
            ucp[d] = alloc_1d_double(NX * NY * NZ_);
        }
    } else if (SW_EQ == Electrolyte) {
        Mem_alloc_charge();
    }
    Mem_alloc_f_particle();
    for (int d = 0; d < DIM - 1; d++) {
        zeta[d] = alloc_1d_double(NX * NY * NZ_);
    }

    u       = (double **)malloc(sizeof(double *) * DIM);
    up      = (double **)malloc(sizeof(double *) * DIM);
    work_v3 = (double **)malloc(sizeof(double *) * DIM);

    for (int d = 0; d < DIM; d++) {
        u[d]       = alloc_1d_double(NX * NY * NZ_);
        up[d]      = alloc_1d_double(NX * NY * NZ_);
        work_v3[d] = alloc_1d_double(NX * NY * NZ_);
    }

    work_v2 = (double **)malloc(sizeof(double *) * (DIM - 1));
    for (int d = 0; d < DIM - 1; d++) {
        work_v2[d] = alloc_1d_double(NX * NY * NZ_);
    }

    phi             = alloc_1d_double(NX * NY * NZ_);
    phi_p           = alloc_1d_double(NX * NY * NZ_);
    phi_sum         = alloc_1d_double(NX * NY * NZ_);
    phi_wall        = alloc_1d_double(NX * NY * NZ_);
    rhop            = alloc_1d_double(NX * NY * NZ_);
    work_v1         = alloc_1d_double(NX * NY * NZ_);
    Hydro_force     = alloc_1d_double(NX * NY * NZ_);
    Hydro_force_new = alloc_1d_double(NX * NY * NZ_);

    shear_rate_field = alloc_1d_double(NX * NY * NZ_);

    if (SW_EQ == Navier_Stokes_FDM || SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM ||
        SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
        Mem_alloc_fdm();
#ifdef _LIS_SOLVER
        Mem_alloc_lis();
#else
        Mem_alloc_matrix_solver();
#endif
    }
}

inline void init_threads() {
#ifdef _OPENMP
    {
        int nthreads, tid, procs, maxt, inpar, dynamic, nested;
#pragma omp parallel private(nthreads, tid)
        {
            tid = omp_get_thread_num();
            if (tid == 0) {
                cerr << "#" << endl;
                cerr << "# OMP RUNTIME : " << endl;
                cerr << "# Number of processors        : " << omp_get_thread_num() << endl;
                cerr << "# Number of threads           : " << omp_get_num_threads() << endl;
                cerr << "# Max OMP threads             : " << omp_get_max_threads() << endl;
#ifdef _FFT_IMKL
                cerr << "# Max MKL threads             : " << mkl_get_max_threads() << endl;
#endif
                cerr << "# Dynamic thread enabled?     : " << omp_get_dynamic() << endl;
                cerr << "# Nested parallelism enabled? : " << omp_get_nested() << endl;
                cerr << "#" << endl;
            }
        }
    }
#endif
}

int main(int argc, char *argv[]) {
#ifndef NDEBUG
    cerr << "###########################" << endl;
    cerr << "#  " << endl;
    cerr << "# built by debug mode " << endl;
    cerr << "# (NDEBUG is not defined in sp_3d_ns.h)" << endl;
    cerr << "#  " << endl;
    cerr << "###########################" << endl;
#endif
    init_threads();
    wall_timer global_timer, block_timer;

    global_timer.start();
    block_timer.start();

    // PHASE_SEPARATION = 0;
    // VISCOSITY_CHANGE = 0;

    if (argc > 0) {
        file_get(argc, argv);
        Gourmet_file_io(In_udf, Out_udf, Sum_udf, Def_udf, Ctrl_udf, Res_udf);
    }

    // Main time evolution type
    int U2M = 0;
    switch (SW_EQ) {
        case Navier_Stokes:
            fprintf(stderr, "# Evolution type Navier_Stokes (Spectral Method)\n");
            Time_evolution = Time_evolution_hydro;
            break;
        case Shear_Navier_Stokes:
            fprintf(stderr, "# Evolution type Shear_Navier_Stokes (Spectral Method)\n");
            Time_evolution = Time_evolution_hydro;
            break;
        case Electrolyte:
            fprintf(stderr, "# Evolution type Electrolyte (Spectral Method)\n");
            Time_evolution = Time_evolution_hydro;
            break;
        case Shear_Navier_Stokes_Lees_Edwards:
            fprintf(stderr, "# Evolution type Shear_Navier_Stokes_Lees_Edwards (Spectral Method)\n");
            Time_evolution = Time_evolution_hydro_OBL;
            break;
        case Navier_Stokes_FDM:
            U2M = 1;
            if (SW_NSST == explicit_scheme) {
                fprintf(stderr, "# Evolution type Navier_Stokes (Finite Difference Method, explicit scheme)\n");
            } else if (SW_NSST == implicit_scheme) {
                fprintf(stderr, "# Evolution type Navier_Stokes (Finite Difference Method, implicit scheme)\n");
            }
            Time_evolution_fdm = Time_evolution_hydro_fdm;
            break;
        case Navier_Stokes_Cahn_Hilliard_FDM:
            U2M = 1;
            if (SW_NSST == explicit_scheme) {
                fprintf(stderr, "# Evolution type Navier_Stokes (Finite Difference Method, explicit scheme)\n");
            } else if (SW_NSST == implicit_scheme) {
                fprintf(stderr, "# Evolution type Navier_Stokes (Finite Difference Method, implicit scheme)\n");
            }
            if (SW_CHST == explicit_scheme) {
                fprintf(stderr, "# Evolution type Cahn_Hilliard (Finite Difference Method, explicit scheme)\n");
            } else if (SW_CHST == implicit_scheme) {
                fprintf(stderr, "# Evolution type Cahn_Hilliard (Finite Difference Method, implicit scheme)\n");
            }
            Time_evolution_fdm = Time_evolution_hydro_fdm;
            break;
        case Shear_Navier_Stokes_Lees_Edwards_FDM:
            U2M = 1;
            if (SW_NSST == explicit_scheme) {
                fprintf(
                    stderr,
                    "# Evolution type Shear_Navier_Stokes_Lees_Edwards (Finite Difference Method, explicit scheme)\n");
            } else if (SW_NSST == implicit_scheme) {
                fprintf(
                    stderr,
                    "# Evolution type Shear_Navier_Stokes_Lees_Edwards (Finite Difference Method, implicit scheme)\n");
            }
            Time_evolution_fdm = Time_evolution_hydro_OBL_fdm;
            break;
        case Shear_NS_LE_CH_FDM:
            U2M = 1;
            if (SW_NSST == explicit_scheme) {
                fprintf(
                    stderr,
                    "# Evolution type Shear_Navier_Stokes_Lees_Edwards (Finite Difference Method, explicit scheme)\n");
            } else if (SW_NSST == implicit_scheme) {
                fprintf(
                    stderr,
                    "# Evolution type Shear_Navier_Stokes_Lees_Edwards (Finite Difference Method, implicit scheme)\n");
            }
            if (SW_CHST == explicit_scheme) {
                fprintf(stderr, "# Evolution type Cahn_Hilliard (Finite Difference Method, explicit scheme)\n");
            } else if (SW_CHST == implicit_scheme) {
                fprintf(stderr, "# Evolution type Cahn_Hilliard (Finite Difference Method, implicit scheme)\n");
            }
            Time_evolution_fdm = Time_evolution_hydro_OBL_fdm;
            break;
    }
    if (VISCOSITY_CHANGE) {
        fprintf(stderr, "# Different viscosity fluids -> (ETA_A, ETA_B) = (%f, %f)\n", ETA_A, ETA_B);
    }

    MT_seed(GIVEN_SEED, 0);
    // MT_seed(RANDOM_SEED,0);
    Init_fft();
    Init_Transform_obl();

    XYaspect  = (double)NX / (double)NY;
    sreff_old = 0.;

    double uk_dc[DIM];

    double **zeta;
    zeta = (double **)malloc(sizeof(double *) * (DIM - 1));
    Mem_alloc_var(zeta);

    static CTime jikan = {0, 0.0, DT, DT * 0.5, DT, DT * 0.5};

    if (U2M) {
#ifdef _LIS_SOLVER
        Init_lis(argc, argv);
#else
        if (SW_NSST == implicit_scheme) {
            Init_ns();
        }
        if (SW_CHST == implicit_scheme) {
            Init_ch();
        }
#endif
    }

    Particle *particles = new Particle[Particle_Number];
    if (Particle_Number > 0) {
        Init_Particle(particles);
        if ((SW_PT == chain) && !(DISTRIBUTION == user_specify)) {
            Init_Chain(particles);
        } else if ((SW_PT == rigid && SW_QUINCKE == QUINCKE_OFF) && !(DISTRIBUTION == user_specify)) {
            Init_Rigid(particles);
        }
    }
    if (SW_MULTIPOLE == MULTIPOLE_ON) init_ewald_sum(LX, LY, LZ, Particle_Number);

    Init_Wall(phi_wall);
    Init_output(particles);
    Init_zeta_k(zeta, uk_dc);
    {
        Reset_phi_u(phi, up);
        if (SW_WALL != NO_WALL) {
            Copy_v1(phi_sum, phi_wall);
        } else {
            Reset_phi(phi_sum);
        }
        Make_phi_particle_sum(phi, phi_sum, particles);
        Make_phi_p(phi_p, phi, phi_wall);
        Make_u_particle_sum(up, phi_sum, particles);
        Zeta_k2u(zeta, uk_dc, u);

        Make_f_particle_dt_sole(f_particle, u, up, phi);
        Add_f_particle(u, f_particle);
        U2zeta_k(zeta, uk_dc, u);

        if (1) {
            for (int d = 0; d < DIM; d++) {
                uk_dc[d] = 0.0;
            }
        }
    }
    if (RESUMED == 0) {
        if (U2M) {
            // xdmf output
            xdmf_output(jikan);
        }
    }
    if (SW_EQ == Electrolyte) {
        Init_rho_ion(Concentration, particles, jikan);
    }
    if (PHASE_SEPARATION) {
        Init_phase_separation(phi, psi);
    }
    //  return EXIT_SUCCESS;

    Show_parameter(particles);
    Show_output_parameter();

    if ((SW_EQ == Shear_Navier_Stokes) || (SW_EQ == Shear_Navier_Stokes_Lees_Edwards) ||
        (SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM) || (SW_EQ == Shear_NS_LE_CH_FDM)) {
        Mean_shear_stress(INIT, stderr, particles, jikan, Shear_rate_eff);
    } else if (SW_EQ == Electrolyte) {
        Electrolyte_free_energy(INIT, stderr, particles, Concentration, jikan);
    }

    fprintf(stderr, "# Initialization time (s): %12.3f\n", block_timer.stop());
    block_timer.start();

    //////////////////////////////////////////////////////////
    int resumed_ts = 0;
    if (RESUMED) resumed_ts = last_ts;
    for (jikan.ts = resumed_ts; jikan.ts <= MSTEP; jikan.ts++) {
        int resumed_and_1st_loop = 0;

        if (RESUMED && (jikan.ts == resumed_ts)) {
            resumed_and_1st_loop = 1;
        }
        if (jikan.ts % GTS == 0) {
            if (!resumed_and_1st_loop) {
                if (SW_OUTFORMAT != OUT_NONE) {  // Output field & particle data
                    Output_open_frame();
                    if (SW_EQ != Electrolyte) {
                        Output_field_data(zeta, uk_dc, particles, jikan);
                    } else if (SW_EQ == Electrolyte) {
                        Output_charge_field_data(zeta, uk_dc, Concentration, particles, jikan);
                    }
                    Output_particle_data(particles, jikan);
                    Output_close_frame();
                }

                if (SW_UDF) {  // Output_UDF
                    if (PHASE_SEPARATION) {
                        if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
                            Output_udf_fdm_phase_separation(ufout, psi, particles, jikan);
                        } else if (SW_EQ == Shear_NS_LE_CH_FDM) {
                            A_oblique2a_out(psi, work_v1);
                            Output_udf_fdm_phase_separation(ufout, work_v1, particles, jikan);
                        }
                    } else {
                        Output_udf(ufout, zeta, uk_dc, particles, jikan);
                    }
                }

                if (PHASE_SEPARATION) {
                    if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
                       Output_hdf5_sca("orderparam", "PSI", psi, jikan.ts / GTS);
                    } else if (SW_EQ == Shear_NS_LE_CH_FDM) {
                        A_oblique2a_out(psi, work_v1);
                        Output_hdf5_sca("orderparam", "PSI", work_v1, jikan.ts / GTS);
                    }
                }

                if (U2M) {
                    if (Particle_Number > 0) {
                        // phi is stored in cartesian
                        Output_hdf5_sca("particle", "PHI", phi, jikan.ts / GTS);

                        if (Particle_Number == 1) {
                            Output_hdf5_particle_single("particle_data", particles, jikan.ts / GTS);
                        } else {
                            Output_hdf5_particle("particle_data", particles, jikan.ts / GTS);
                        }
                    }

                    /* velocity u field*/
                    /*
if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards || SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM || SW_EQ
== Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) { if (SW_EQ ==
Shear_Navier_Stokes_Lees_Edwards) { Cpy_v3(work_v3, u); U_k2u(work_v3);
calc_shear_rate_field(work_v3, shear_rate_field); } else if (SW_EQ ==
Navier_Stokes_Cahn_Hilliard_FDM || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ ==
Shear_NS_LE_CH_FDM) { calc_shear_rate_field(u, shear_rate_field);
                            }
                            A_oblique2a_out(shear_rate_field, work_v1);
                            Output_hdf5_sca("shear_rate", "GAMMA", work_v1, jikan.ts / GTS);
                    }
                    */

                    if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
                        Cpy_v3(w_v3, u);
                        Transform_obl_u(w_v3, oblique2cartesian);

                        Output_hdf5_vec("velocity", "UX", "UY", "UZ", w_v3, jikan.ts / GTS);
                    } else {
                        Output_hdf5_vec("velocity", "UX", "UY", "UZ", u, jikan.ts / GTS);
                    }
                }
                if (SW_EQ == Electrolyte) {
                    Electrolyte_free_energy(SHOW, stderr, particles, Concentration, jikan);
                }
                if (jikan.ts != resumed_ts) {
                    double block_time  = block_timer.stop();
                    double global_time = global_timer.stop();
                    fprintf(stderr,
                            "# Step: %9d/%9d\t  Block time (m): %8.3f\t Global time (m): %8.3f/%8.3f\n",
                            jikan.ts,
                            MSTEP,
                            block_time / 60.0,
                            global_time / 60.0,
                            ((double)(MSTEP - jikan.ts + 1)) / ((double)GTS) * block_time / 60.0);
                    fflush(stderr);
                    block_timer.start();
                }
            }
        }
        if (U2M) {
            Time_evolution_fdm(u, Pressure, f_particle, particles, jikan);
        } else {
            Time_evolution(zeta, uk_dc, f_particle, particles, jikan);
        }
        jikan.time += jikan.dt_fluid;

        if (SW_EQ == Shear_Navier_Stokes) {
            Shear_rate_eff = Update_strain(Shear_strain_realized, jikan, zeta, uk_dc, u);
            Mean_shear_stress(SHOW, stderr, particles, jikan, Shear_rate_eff);
        } else if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM ||
                   SW_EQ == Shear_NS_LE_CH_FDM) {
            Shear_strain_realized += Shear_rate_eff * jikan.dt_fluid;
            Mean_shear_stress(SHOW, stderr, particles, jikan, Shear_rate_eff);
        }

        if (jikan.ts == MSTEP) {
            if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
                Save_Restart_udf_fdm_phase_separation(u, u_o, psi, psi_o, stress_o, particles, jikan);
            } else if (SW_EQ == Navier_Stokes_FDM || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM) {
                Save_Restart_udf_fdm(u, u_o, particles, jikan);
            } else {
                Save_Restart_udf(zeta, uk_dc, particles, jikan, Concentration);
            }
        }

        if (resumed_and_1st_loop) {
            if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
                Force_restore_parameters_fdm_phase_separation(u, u_o, psi, psi_o, stress_o, particles, jikan);
                U2advection(u, adv);
                U2laplacian(u, lap);
                U2advection(u_o, adv_o);
                Reset_phi(phi);
                Reset_phi(phi_sum);
                Make_phi_particle_sum(phi, phi_sum, particles);
                if (PHASE_SEPARATION) {
                    if (SW_WALL != NO_WALL){
                        Calc_cp_wall(phi, phi_p, phi_wall, psi, cp);
                    }
                    else{
                        Calc_cp(phi,psi,cp);
                    }
                    Cp2stress(cp, psi, stress);
                }
            } else if (SW_EQ == Navier_Stokes_FDM || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM) {
                Force_restore_parameters_fdm(u, u_o, particles, jikan);
                U2advection(u, adv);
                U2laplacian(u, lap);
                U2advection(u_o, adv_o);
                Reset_phi(phi);
                Reset_phi(phi_sum);
                Make_phi_particle_sum(phi, phi_sum, particles);
            } else {
                Force_restore_parameters(zeta, uk_dc, particles, jikan, Concentration);
            }
            delete ufin;
            fprintf(stderr, "############################ Parameters are restored.\n");
            if (U2M) {
                // xdmf output
                xdmf_output(jikan);
            }
        }
    }
    if (SW_MULTIPOLE == MULTIPOLE_ON) {
        free_ewald_sum();
        delete ewald_sum;
    }

    if (SW_UDF) {
        ufout->write();
        delete ufout;
        fprintf(stderr, "#%s end.\n", Out_udf);
    }
    {
        // Always write restart file
        ufres->write();
        delete ufres;
        fprintf(stderr, "#%s end.\n", Res_udf);
    }
    Free_output();

    {
        double global_time = global_timer.stop();
        fprintf(stderr, "#Simulation has ended!\n");
        fprintf(stderr, "#Total Running Time (s): %10.2f\n", global_time);
        fprintf(stderr, "#                   (m): %10.2f\n", global_time / 60.0);
        fprintf(stderr, "#                   (h): %10.2f\n", global_time / 3600.0);

        global_time /= (double)(MSTEP - resumed_ts + 1);
        fprintf(stderr, "#Average Step Time  (s): %10.2f\n", global_time);
        fprintf(stderr, "#                   (m): %10.2f\n", global_time / 60.0);
        fprintf(stderr, "#                   (h): %10.2f\n", global_time / 3600.0);
    }

    Free_Transform_obl();
    Free_fft();
    for (int d = 0; d < DIM - 1; d++) {
        free_1d_double(zeta[d]);
    }
    free(zeta);
    delete[] particles;
    if (U2M) {
#ifdef _LIS_SOLVER
        Free_lis();
        lis_finalize();
#else
        Free_matrix_solver();
#endif
        Free_fdm();
        free_1d_double(shear_rate_field);
    }
    return EXIT_SUCCESS;
}
