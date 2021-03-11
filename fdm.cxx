#include "fdm.h"

double **adv;
double **adv_o;
double **lap;
double **lap_o;
double * rhs;
double **u_s;
double **u_o;
double **u_o_cpy;
double * w_v1;
double **w_v3;
double **w_v3_2;
double **w_v3_3;
double * eta_s;
double * shear_rate_field;

double sreff_old;

void NS_solver_slavedEuler_explicit(double **u, double *Pressure, Particle *p, CTime &jikan) {
    if (jikan.ts > 0) {
        Cpy_v3(adv_o, adv);
        if (PHASE_SEPARATION) {
            Cpy_v3(stress_o, stress);
        }
    }

    U2advection(u, adv);
    U2laplacian(u, lap);

    if (jikan.ts < 2) {
        Update_u_adv_euler(u, adv, jikan);
    } else {
        Update_u_adv_ab2(u, adv, adv_o, jikan);
    }

    Update_u_lap_euler(u, lap, jikan);

    if (PHASE_SEPARATION) {
        Calc_cp(phi, psi, cp);
        Cp2stress(cp, psi, stress);

        if (jikan.ts < 2) {
            Update_u_stress_euler(u, stress, jikan);
        } else {
            Update_u_stress_ab2(u, stress, stress_o, jikan);
        }
    }

    if (PHASE_SEPARATION) {
        Set_poisson_rhs_ps(u, adv, lap, stress, rhs, jikan);
    } else {
        Set_poisson_rhs(u, adv, lap, rhs, jikan);
    }

    Solve_poisson_dft(Pressure, rhs);
    Update_u_pressure(u, Pressure, jikan);
}

void NS_solver_slavedEuler_implicit(double **u, double *Pressure, Particle *p, CTime &jikan) {
    Calc_ab2_val(u_s[0], u[0], u_o[0]);
    Calc_ab2_val(u_s[1], u[1], u_o[1]);
    Calc_ab2_val(u_s[2], u[2], u_o[2]);
    Cpy_v3(u_o, u);

    U2advection(u_s, adv);
    U2laplacian(u_s, lap);

    if (PHASE_SEPARATION) {
        // psi_s -> w_v3[0]
        // phi_s -> w_v3[1]
        // phi_sum -> w_v3[2] (dummy working memory)
        // cp_s -> cp
        // stress_s -> stress

        Calc_ab2_val(w_v3[0], psi, psi_o);

        Reset_phi(w_v3[1]);
        Reset_phi(w_v3[2]);
        Make_phi_s(w_v3[1], w_v3[2], p, DX, NP_domain, Sekibun_cell, Ns, jikan);
        if(SW_WALL!=NO_WALL){
            Calc_cp_wall(w_v3[1], phi_p, phi_wall, w_v3[0], cp);
        }
        else{
            Calc_cp(w_v3[1], w_v3[0], cp);
        }
        Cp2stress(cp, w_v3[0], stress);

        if (VISCOSITY_CHANGE) {
            Psi2eta(w_v3[0], eta_s);
            Set_poisson_rhs_viscosity(u, u_s, adv, lap, stress, eta_s, rhs, jikan);

        } else {
            Set_poisson_rhs_ps(u, adv, lap, stress, rhs, jikan);
        }
    } else {
        Set_poisson_rhs(u, adv, lap, rhs, jikan);
    }

    Solve_poisson_dft(Pressure, rhs);

    if (PHASE_SEPARATION) {
        if (VISCOSITY_CHANGE) {
#ifdef _LIS_SOLVER
            CHNS_MAC_solver_implicit_viscosity(u, Pressure, u_s, stress, eta_s, jikan, is_ns, ie_ns);
#else
            CHNS_MAC_solver_implicit_viscosity(u, Pressure, u_s, stress, eta_s, jikan, 0, NX * NY * NZ * DIM);
#endif
        } else {
#ifdef _LIS_SOLVER
            CHNS_MAC_solver_implicit(u, Pressure, u_s, stress, jikan, is_ns, ie_ns);
#else
            CHNS_MAC_solver_implicit(u, Pressure, u_s, stress, jikan, 0, NX * NY * NZ * DIM);
#endif
        }
    } else {
#ifdef _LIS_SOLVER
        NS_MAC_solver_implicit(u, Pressure, u_s, jikan, is_ns, ie_ns);
#else
        NS_MAC_solver_implicit(u, Pressure, u_s, jikan, 0, NX * NY * NZ * DIM);
#endif
    }
}

void NS_solver_slavedEuler_Shear_OBL_explicit(double **u, double *Pressure, Particle *p, CTime &jikan) {
    if (jikan.ts > 0) {
        Cpy_v3(adv_o, adv);
        if (PHASE_SEPARATION) {
            Cpy_v3(stress_o, stress);
        }
    }

    U2advection_OBL(u, adv, degree_oblique);  // adv frame term is included.
    U2laplacian_OBL(u, lap, degree_oblique);

    if (jikan.ts < 2) {
        Update_u_adv_euler(u, adv, jikan);
    } else {
        Update_u_adv_ab2(u, adv, adv_o, jikan);
    }

    Update_u_lap_euler(u, lap, jikan);

    if (PHASE_SEPARATION) {
        Copy_v1(phi_obl, phi);
        A2a_oblique(phi_obl);
        Calc_cp_OBL(phi_obl, psi, cp, degree_oblique);
        Cp2stress_OBL(cp, psi, stress, degree_oblique);

        if (jikan.ts < 2) {
            Update_u_stress_euler(u, stress, jikan);
        } else {
            Update_u_stress_ab2(u, stress, stress_o, jikan);
        }
    }

    if (PHASE_SEPARATION) {
        Set_poisson_rhs_ps(u, adv, lap, stress, rhs, jikan);
    } else {
        Set_poisson_rhs(
            u, adv, lap, rhs, jikan);  // the term arising from advection of the frame is included in advection term
    }

    Solve_poisson_dft(Pressure, rhs);
    Update_u_pressure_OBL(u, Pressure, jikan, degree_oblique);
}

void NS_solver_slavedEuler_Shear_OBL_implicit(double **&u, double *Pressure, Particle *p, CTime &jikan) {
    Calc_ab2_val(u_s[0], u[0], u_o[0]);
    Calc_ab2_val(u_s[1], u[1], u_o[1]);
    Calc_ab2_val(u_s[2], u[2], u_o[2]);

    Cpy_v3(u_o, u);

    double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;
    U2advection_OBL(u_s, adv, gt);
    U2laplacian_OBL(u_s, lap, gt);

    if (PHASE_SEPARATION) {
        // psi_s -> w_v3[0]
        // phi_s -> w_v3[1]
        // phi_sum -> w_v3[2] (dummy working memory)
        // cp_s -> cp

        Calc_ab2_val(w_v3[0], psi, psi_o);
        Reset_phi(w_v3[1]);
        Reset_phi(w_v3[2]);
        Make_phi_s_OBL(
            w_v3[1], w_v3[2], p, DX, NP_domain, Sekibun_cell, Ns, gt, (3. * Shear_rate_eff - sreff_old) / 2., jikan);
        A2a_oblique(w_v3[1]);
        Calc_cp_OBL(w_v3[1], w_v3[0], cp, gt);
        Cp2stress_OBL(cp, w_v3[0], stress, gt);

        if (VISCOSITY_CHANGE) {
            Psi2eta(w_v3[0], eta_s);
            Set_poisson_rhs_viscosity_OBL(u, u_s, adv, lap, stress, eta_s, rhs, jikan);
        } else {
            Set_poisson_rhs_ps(u, adv, lap, stress, rhs, jikan);
        }
    } else {
        Set_poisson_rhs(u, adv, lap, rhs, jikan);
    }
    Update_K2_OBL_hdt(jikan);
    Solve_poisson_dft(Pressure, rhs);
    Update_K2_OBL();

    if (PHASE_SEPARATION) {
        if (VISCOSITY_CHANGE) {
#ifdef _LIS_SOLVER
            CHNS_MAC_solver_implicit_viscosity_OBL(
                u, Pressure, u_s, stress, eta_s, jikan, degree_oblique, is_ns, ie_ns);
#else
            CHNS_MAC_solver_implicit_viscosity_OBL(
                u, Pressure, u_s, stress, eta_s, jikan, degree_oblique, 0, NX * NY * NZ * DIM);
#endif
        } else {
#ifdef _LIS_SOLVER
            CHNS_MAC_solver_implicit_OBL(u, Pressure, u_s, stress, jikan, degree_oblique, is_ns, ie_ns);
#else
            CHNS_MAC_solver_implicit_OBL(u, Pressure, u_s, stress, jikan, degree_oblique, 0, NX * NY * NZ * DIM);
#endif
        }
    } else {
#ifdef _LIS_SOLVER
        NS_MAC_solver_implicit_OBL(u, Pressure, u_s, jikan, degree_oblique, is_ns, ie_ns);
#else
        NS_MAC_solver_implicit_OBL(u, Pressure, u_s, jikan, degree_oblique, 0, NX * NY * NZ * DIM);
#endif
    }
}

void        Set_poisson_rhs(double **u, double **adv_u, double **lap_u, double *s, CTime &jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                for (int d = 0; d < DIM; d++) {
                    w_v3_3[d][im] = adv_u[d][im] - NU * lap_u[d][im];
                }
            }
        }
    }
    Set_poisson_rhs_sub(u, w_v3_3, s, jikan);
}

void Update_u_pressure(double **u, double *dp, CTime &jikan) {
    int    im;
    double dmy = jikan.dt_fluid * IRHO;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;

                double dp_dx = calc_gradient_o2_to_o1(dp, im, 0);
                double dp_dy = calc_gradient_o2_to_o1(dp, im, 1);
                double dp_dz = calc_gradient_o2_to_o1(dp, im, 2);

                u[0][im] -= dmy * dp_dx;
                u[1][im] -= dmy * dp_dy;
                u[2][im] -= dmy * dp_dz;
            }
        }
    }
}

void Update_u_pressure_OBL(double **u, double *dp, CTime &jikan, const double degree_oblique) {
    double dmy = jikan.dt_fluid * IRHO;
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                double dp_dx = calc_gradient_o2_to_o1(dp, im, 0);  // co
                double dp_dy = calc_gradient_o2_to_o1(dp, im, 1);  // co
                double dp_dz = calc_gradient_o2_to_o1(dp, im, 2);  // co

                u[0][im] -= dmy * (((1. + degree_oblique * degree_oblique) * dp_dx) - (degree_oblique * dp_dy));
                u[1][im] -= dmy * (-(degree_oblique * dp_dx) + dp_dy);
                u[2][im] -= dmy * dp_dz;
            }
        }
    }
}

void Update_u_adv_euler(double **u, double **adv_u, CTime &jikan) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] -= jikan.dt_fluid * adv_u[0][im];
                u[1][im] -= jikan.dt_fluid * adv_u[1][im];
                u[2][im] -= jikan.dt_fluid * adv_u[2][im];
            }
        }
    }
}

void Update_u_adv_ab2(double **u, double **adv_u, double **adv_u_old, CTime &jikan) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] -= 0.5 * jikan.dt_fluid * (3. * adv_u[0][im] - adv_u_old[0][im]);
                u[1][im] -= 0.5 * jikan.dt_fluid * (3. * adv_u[1][im] - adv_u_old[1][im]);
                u[2][im] -= 0.5 * jikan.dt_fluid * (3. * adv_u[2][im] - adv_u_old[2][im]);
            }
        }
    }
}

void Update_u_lap_euler(double **u, double **lap_u, CTime &jikan) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] += NU * jikan.dt_fluid * lap_u[0][im];
                u[1][im] += NU * jikan.dt_fluid * lap_u[1][im];
                u[2][im] += NU * jikan.dt_fluid * lap_u[2][im];
            }
        }
    }
}

void Update_u_lap_ab2(double **u, double **lap_u, double **lap_u_old, CTime &jikan) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] += 0.5 * NU * jikan.dt_fluid * (3. * lap_u[0][im] - lap_u_old[0][im]);
                u[1][im] += 0.5 * NU * jikan.dt_fluid * (3. * lap_u[1][im] - lap_u_old[1][im]);
                u[2][im] += 0.5 * NU * jikan.dt_fluid * (3. * lap_u[2][im] - lap_u_old[2][im]);
            }
        }
    }
}

void Update_p(double *p, double *dp) {
    // correction of pressure
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                p[im] += dp[im];
            }
        }
    }
}

void U2advection(double **u, double **adv_u) {
    const double INV_2DX = 1. / (2. * DX);
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                // periodic boundary condition
                int ip1 = adj(1, i, NX);
                int jp1 = adj(1, j, NY);
                int kp1 = adj(1, k, NZ);

                int im1 = adj(-1, i, NX);
                int jm1 = adj(-1, j, NY);
                int km1 = adj(-1, k, NZ);

                // set adjacent meshes
                int im_ip1 = ijk2im(ip1, j, k);
                int im_jp1 = ijk2im(i, jp1, k);
                int im_kp1 = ijk2im(i, j, kp1);

                int im_im1 = ijk2im(im1, j, k);
                int im_jm1 = ijk2im(i, jm1, k);
                int im_km1 = ijk2im(i, j, km1);

                double dux_dx = (u[0][im_ip1] - u[0][im_im1]) * INV_2DX;
                double dux_dy = (u[0][im_jp1] - u[0][im_jm1]) * INV_2DX;
                double dux_dz = (u[0][im_kp1] - u[0][im_km1]) * INV_2DX;

                double duy_dx = (u[1][im_ip1] - u[1][im_im1]) * INV_2DX;
                double duy_dy = (u[1][im_jp1] - u[1][im_jm1]) * INV_2DX;
                double duy_dz = (u[1][im_kp1] - u[1][im_km1]) * INV_2DX;

                double duz_dx = (u[2][im_ip1] - u[2][im_im1]) * INV_2DX;
                double duz_dy = (u[2][im_jp1] - u[2][im_jm1]) * INV_2DX;
                double duz_dz = (u[2][im_kp1] - u[2][im_km1]) * INV_2DX;

                // non-conservation form
                adv_u[0][im] = u[0][im] * dux_dx + u[1][im] * dux_dy + u[2][im] * dux_dz;
                adv_u[1][im] = u[0][im] * duy_dx + u[1][im] * duy_dy + u[2][im] * duy_dz;
                adv_u[2][im] = u[0][im] * duz_dx + u[1][im] * duz_dy + u[2][im] * duz_dz;
            }
        }
    }
}

void U2laplacian(double **u, double **lap_u) {
    const double INV_DX2 = 1. / (DX * DX);
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                // periodic boundary condition
                int ip1 = adj(1, i, NX);
                int jp1 = adj(1, j, NY);
                int kp1 = adj(1, k, NZ);

                int im1 = adj(-1, i, NX);
                int jm1 = adj(-1, j, NY);
                int km1 = adj(-1, k, NZ);

                // set adjacent meshes
                int im_ip1 = ijk2im(ip1, j, k);
                int im_jp1 = ijk2im(i, jp1, k);
                int im_kp1 = ijk2im(i, j, kp1);

                int im_im1 = ijk2im(im1, j, k);
                int im_jm1 = ijk2im(i, jm1, k);
                int im_km1 = ijk2im(i, j, km1);

                double d2ux_dx2 = (u[0][im_ip1] - 2. * u[0][im] + u[0][im_im1]) * INV_DX2;
                double d2ux_dy2 = (u[0][im_jp1] - 2. * u[0][im] + u[0][im_jm1]) * INV_DX2;
                double d2ux_dz2 = (u[0][im_kp1] - 2. * u[0][im] + u[0][im_km1]) * INV_DX2;

                double d2uy_dx2 = (u[1][im_ip1] - 2. * u[1][im] + u[1][im_im1]) * INV_DX2;
                double d2uy_dy2 = (u[1][im_jp1] - 2. * u[1][im] + u[1][im_jm1]) * INV_DX2;
                double d2uy_dz2 = (u[1][im_kp1] - 2. * u[1][im] + u[1][im_km1]) * INV_DX2;

                double d2uz_dx2 = (u[2][im_ip1] - 2. * u[2][im] + u[2][im_im1]) * INV_DX2;
                double d2uz_dy2 = (u[2][im_jp1] - 2. * u[2][im] + u[2][im_jm1]) * INV_DX2;
                double d2uz_dz2 = (u[2][im_kp1] - 2. * u[2][im] + u[2][im_km1]) * INV_DX2;

                lap_u[0][im] = d2ux_dx2 + d2ux_dy2 + d2ux_dz2;
                lap_u[1][im] = d2uy_dx2 + d2uy_dy2 + d2uy_dz2;
                lap_u[2][im] = d2uz_dx2 + d2uz_dy2 + d2uz_dz2;
            }
        }
    }
}

void U2advection_OBL(double **u, double **adv_u, const double degree_oblique) {
    const double INV_2DX = 1. / (2. * DX);
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                // periodic boundary condition
                int ip1 = adj(1, i, NX);
                int jp1 = adj(1, j, NY);
                int kp1 = adj(1, k, NZ);

                int im1 = adj(-1, i, NX);
                int jm1 = adj(-1, j, NY);
                int km1 = adj(-1, k, NZ);

                // set adjacent meshes
                int im_ip1 = ijk2im(ip1, j, k);
                int im_jp1 = ijk2im(i, jp1, k);
                int im_kp1 = ijk2im(i, j, kp1);

                int im_im1 = ijk2im(im1, j, k);
                int im_jm1 = ijk2im(i, jm1, k);
                int im_km1 = ijk2im(i, j, km1);

                double dux_dx = (u[0][im_ip1] - u[0][im_im1]) * INV_2DX;
                double dux_dy = (u[0][im_jp1] - u[0][im_jm1]) * INV_2DX;
                double dux_dz = (u[0][im_kp1] - u[0][im_km1]) * INV_2DX;

                double duy_dx = (u[1][im_ip1] - u[1][im_im1]) * INV_2DX;
                double duy_dy = (u[1][im_jp1] - u[1][im_jm1]) * INV_2DX;
                double duy_dz = (u[1][im_kp1] - u[1][im_km1]) * INV_2DX;

                double duz_dx = (u[2][im_ip1] - u[2][im_im1]) * INV_2DX;
                double duz_dy = (u[2][im_jp1] - u[2][im_jm1]) * INV_2DX;
                double duz_dz = (u[2][im_kp1] - u[2][im_km1]) * INV_2DX;

                // non-conservation form
                adv_u[0][im] = u[0][im] * dux_dx + u[1][im] * dux_dy + u[2][im] * dux_dz;
                adv_u[1][im] = u[0][im] * duy_dx + u[1][im] * duy_dy + u[2][im] * duy_dz;
                adv_u[2][im] = u[0][im] * duz_dx + u[1][im] * duz_dy + u[2][im] * duz_dz;
            }
        }
    }

    // advection of oblique frames
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                adv_u[0][im] += 2. * Shear_rate_eff * u[1][im];  // contra
            }
        }
    }
}

void U2laplacian_OBL(double **u, double **lap_u, const double degree_oblique) {
    const double INV_DX2  = 1. / (DX * DX);
    const double INV_4DX2 = 1. / (4. * DX * DX);
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                int ip1 = adj(1, i, NX);
                int jp1 = adj(1, j, NY);
                int kp1 = adj(1, k, NZ);

                int im1 = adj(-1, i, NX);
                int jm1 = adj(-1, j, NY);
                int km1 = adj(-1, k, NZ);

                // set adjacent meshes
                int im_ip1 = ijk2im(ip1, j, k);
                int im_jp1 = ijk2im(i, jp1, k);
                int im_kp1 = ijk2im(i, j, kp1);

                int im_im1 = ijk2im(im1, j, k);
                int im_jm1 = ijk2im(i, jm1, k);
                int im_km1 = ijk2im(i, j, km1);

                int im_ip1_jp1 = ijk2im(ip1, jp1, k);
                int im_ip1_jm1 = ijk2im(ip1, jm1, k);
                int im_im1_jp1 = ijk2im(im1, jp1, k);
                int im_im1_jm1 = ijk2im(im1, jm1, k);

                double d2ux_dx2 = (u[0][im_ip1] - 2. * u[0][im] + u[0][im_im1]) * INV_DX2;
                double d2ux_dy2 = (u[0][im_jp1] - 2. * u[0][im] + u[0][im_jm1]) * INV_DX2;
                double d2ux_dz2 = (u[0][im_kp1] - 2. * u[0][im] + u[0][im_km1]) * INV_DX2;

                double d2uy_dx2 = (u[1][im_ip1] - 2. * u[1][im] + u[1][im_im1]) * INV_DX2;
                double d2uy_dy2 = (u[1][im_jp1] - 2. * u[1][im] + u[1][im_jm1]) * INV_DX2;
                double d2uy_dz2 = (u[1][im_kp1] - 2. * u[1][im] + u[1][im_km1]) * INV_DX2;

                double d2uz_dx2 = (u[2][im_ip1] - 2. * u[2][im] + u[2][im_im1]) * INV_DX2;
                double d2uz_dy2 = (u[2][im_jp1] - 2. * u[2][im] + u[2][im_jm1]) * INV_DX2;
                double d2uz_dz2 = (u[2][im_kp1] - 2. * u[2][im] + u[2][im_km1]) * INV_DX2;

                double d2ux_dxdy =
                    (u[0][im_ip1_jp1] - u[0][im_ip1_jm1] - u[0][im_im1_jp1] + u[0][im_im1_jm1]) * INV_4DX2;
                double d2uy_dxdy =
                    (u[1][im_ip1_jp1] - u[1][im_ip1_jm1] - u[1][im_im1_jp1] + u[1][im_im1_jm1]) * INV_4DX2;
                double d2uz_dxdy =
                    (u[2][im_ip1_jp1] - u[2][im_ip1_jm1] - u[2][im_im1_jp1] + u[2][im_im1_jm1]) * INV_4DX2;

                lap_u[0][im] = (1. + degree_oblique * degree_oblique) * d2ux_dx2 - (2. * degree_oblique * d2ux_dxdy) +
                               d2ux_dy2 + d2ux_dz2;
                lap_u[1][im] = (1. + degree_oblique * degree_oblique) * d2uy_dx2 - (2. * degree_oblique * d2uy_dxdy) +
                               d2uy_dy2 + d2uy_dz2;
                lap_u[2][im] = (1. + degree_oblique * degree_oblique) * d2uz_dx2 - (2. * degree_oblique * d2uz_dxdy) +
                               d2uz_dy2 + d2uz_dz2;
            }
        }
    }
}

void Mem_alloc_fdm(void) {
    adv   = alloc_2d_double(DIM, NX * NY * NZ_);
    adv_o = calloc_2d_double(DIM, NX * NY * NZ_);
    lap   = alloc_2d_double(DIM, NX * NY * NZ_);
    lap_o = calloc_2d_double(DIM, NX * NY * NZ_);

    u_s     = alloc_2d_double(DIM, NX * NY * NZ_);
    u_o     = calloc_2d_double(DIM, NX * NY * NZ_);
    u_o_cpy = calloc_2d_double(DIM, NX * NY * NZ_);
    rhs     = alloc_1d_double(NX * NY * NZ_);

    w_v1   = alloc_1d_double(NX * NY * NZ_);
    w_v3   = alloc_2d_double(DIM, NX * NY * NZ_);
    w_v3_2 = alloc_2d_double(DIM, NX * NY * NZ_);
    w_v3_3 = alloc_2d_double(DIM, NX * NY * NZ_);

    if (PHASE_SEPARATION) {
        stress   = alloc_2d_double(DIM, NX * NY * NZ_);
        stress_o = alloc_2d_double(DIM, NX * NY * NZ_);
        cp       = alloc_1d_double(NX * NY * NZ_);
        psi      = alloc_1d_double(NX * NY * NZ_);
        psi_o    = alloc_1d_double(NX * NY * NZ_);
        psicp    = alloc_1d_double(NX * NY * NZ_);
        psicp_o  = alloc_1d_double(NX * NY * NZ_);
        phi_obl  = alloc_1d_double(NX * NY * NZ_);
    }

    if (VISCOSITY_CHANGE) {
        eta_s = alloc_1d_double(NX * NY * NZ_);
    }
}

void Free_fdm(void) {
    free_2d_double(adv);
    free_2d_double(adv_o);
    free_2d_double(lap);
    free_2d_double(lap_o);
    free_2d_double(u_s);
    free_2d_double(u_o);
    free_2d_double(u_o_cpy);
    free_1d_double(rhs);
    free_1d_double(w_v1);
    free_2d_double(w_v3);
    free_2d_double(w_v3_2);
    free_2d_double(w_v3_3);

    if (PHASE_SEPARATION) {
        free_2d_double(stress);
        free_2d_double(stress_o);
        free_1d_double(cp);
        free_1d_double(psi);
        free_1d_double(psi_o);
        free_1d_double(psicp);
        free_1d_double(psicp_o);
        free_1d_double(phi_obl);
    }

    if (VISCOSITY_CHANGE) {
        free_1d_double(eta_s);
    }
}

void        calc_shear_rate_field(double **u, double *shear_rate_field) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im               = (i * NY * NZ_) + (j * NZ_) + k;
                shear_rate_field[im] = calc_gamma(u, im);
            }
        }
    }
}