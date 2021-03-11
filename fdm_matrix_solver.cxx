#include "fdm_matrix_solver.h"

#ifdef _LIS_SOLVER
LIS_INT *   ptr_ns, *idx_ns_csr;
LIS_SCALAR *val_ns_csr;
LIS_MATRIX  A_ns;
LIS_VECTOR  b_ns, x_ns;
LIS_SOLVER  lis_solver_ns;
LIS_INT     is_ns, ie_ns;

LIS_INT *   ptr_ch, *idx_ch_csr;
LIS_SCALAR *val_ch_csr;
LIS_MATRIX  A_ch;
LIS_VECTOR  b_ch, x_ch;
LIS_SOLVER  lis_solver_ch;
LIS_INT     is_ch, ie_ch;
#else
int *         idx_ns_csr;
int *         ptr_ns;
double *      val_ns_csr;
double *      b_ns;
work_bicgstab wm_ns;

int *         idx_ch_csr;
int *         ptr_ch;
double *      val_ch_csr;
double *      b_ch;
work_bicgstab wm_ch;
#endif

void NS_MAC_solver_implicit(double **u, double *pressure, double **u_s, const CTime &jikan, int is_ns, int ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const int    nval     = NX * NY * NZ * DIM;
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 7 * (idx - is_ns);

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = 1. / jikan.dt_fluid + 3. * NU / DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);
        val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - NU * INV_2DX2;

        ptr_ns[idx - is_ns + 1] = idx2 + 7;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 * (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) / _2DX +
                                  u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) / _2DX +
                                  u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)]) / _2DX);

        double ps = 0;
        switch (d) {
            case 0:
                ps = pressure[im] - pressure[ijk2im(im1, j, k)] + pressure[ijk2im(i, jm1, k)] -
                     pressure[ijk2im(im1, jm1, k)] + pressure[ijk2im(i, j, km1)] - pressure[ijk2im(im1, j, km1)] +
                     pressure[ijk2im(i, jm1, km1)] - pressure[ijk2im(im1, jm1, km1)];
                break;
            case 1:
                ps = pressure[im] - pressure[ijk2im(i, jm1, k)] + pressure[ijk2im(im1, j, k)] -
                     pressure[ijk2im(im1, jm1, k)] + pressure[ijk2im(i, j, km1)] - pressure[ijk2im(i, jm1, km1)] +
                     pressure[ijk2im(im1, j, km1)] - pressure[ijk2im(im1, jm1, km1)];
                break;
            case 2:
                ps = pressure[im] - pressure[ijk2im(i, j, km1)] + pressure[ijk2im(im1, j, k)] -
                     pressure[ijk2im(im1, j, km1)] + pressure[ijk2im(i, jm1, k)] - pressure[ijk2im(i, jm1, km1)] +
                     pressure[ijk2im(im1, jm1, k)] - pressure[ijk2im(im1, jm1, km1)];
                break;
            default:
                break;
        }
        double pressure_term = -IRHO * ps / (4. * DX);

        double viscosity_term = 0.5 * NU * calc_laplacian(u[d], im);

        double bs = u[d][im] / jikan.dt_fluid + adv_term + pressure_term + viscosity_term;
#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 7, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
    // if (jikan.ts % 1000 == 0) {
    //	lis_solver_get_iter(lis_solver_ns, &iter);
    //	printf("%d NS matrix solver iter = %d\n", jikan.ts, iter);
    //}
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void CHNS_MAC_solver_implicit(double **    u,
                              double *     pressure,
                              double **    u_s,
                              double **    stress_s,
                              const CTime &jikan,
                              int          is_ns,
                              int          ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const int    nval     = NX * NY * NZ * DIM;
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 7 * (idx - is_ns);

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = 1. / jikan.dt_fluid + 3. * NU / DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);
        val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - NU * INV_2DX2;

        ptr_ns[idx - is_ns + 1] = idx2 + 7;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 * (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) / _2DX +
                                  u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) / _2DX +
                                  u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)]) / _2DX);

        double pressure_term  = -IRHO * calc_gradient_o2_to_o1(pressure, im, d);
        double viscosity_term = 0.5 * NU * calc_laplacian(u[d], im);
        double stress_term    = -stress[d][im];
        double bs             = u[d][im] / jikan.dt_fluid + adv_term + pressure_term + viscosity_term + stress_term;
#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 7, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void CHNS_MAC_solver_implicit_viscosity(double **    u,
                                        double *     pressure,
                                        double **    u_s,
                                        double **    stress_s,
                                        double *     eta_s,
                                        const CTime &jikan,
                                        int          is_ns,
                                        int          ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_2DX  = 1. / _2DX;
    const double INV_DX2  = 1. / DX2;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const double INV_DT   = 1. / jikan.dt_fluid;
    const int    nval     = NX * NY * NZ * DIM;
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 11 * (idx - is_ns);

        double nu = eta_s[im] * IRHO;

        double nu_x = calc_gradient_o1_to_o1(eta_s, im, 0) * IRHO * INV_2DX;
        double nu_y = calc_gradient_o1_to_o1(eta_s, im, 1) * IRHO * INV_2DX;
        double nu_z = calc_gradient_o1_to_o1(eta_s, im, 2) * IRHO * INV_2DX;

        double h_nu_x = 0.5 * nu_x;
        double h_nu_y = 0.5 * nu_y;
        double h_nu_z = 0.5 * nu_z;

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = INV_DT + 3. * nu * INV_DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);

        switch (d) {
            case 0:
                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * INV_2DX2 - nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * INV_2DX2 + nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - h_nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + h_nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - h_nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + h_nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(ip1, j, k, 1);
                val_ns_csr[idx2 + 7] = -h_nu_y;

                idx_ns_csr[idx2 + 8] = ijkd2idx(im1, j, k, 1);
                val_ns_csr[idx2 + 8] = h_nu_y;

                idx_ns_csr[idx2 + 9] = ijkd2idx(ip1, j, k, 2);
                val_ns_csr[idx2 + 9] = -h_nu_z;

                idx_ns_csr[idx2 + 10] = ijkd2idx(im1, j, k, 2);
                val_ns_csr[idx2 + 10] = h_nu_z;
                break;

            case 1:
                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * INV_2DX2 - h_nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * INV_2DX2 + h_nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - h_nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + h_nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(i, jp1, k, 0);
                val_ns_csr[idx2 + 7] = -h_nu_x;

                idx_ns_csr[idx2 + 8] = ijkd2idx(i, jm1, k, 0);
                val_ns_csr[idx2 + 8] = h_nu_x;

                idx_ns_csr[idx2 + 9] = ijkd2idx(i, jp1, k, 2);
                val_ns_csr[idx2 + 9] = -h_nu_z;

                idx_ns_csr[idx2 + 10] = ijkd2idx(i, jm1, k, 2);
                val_ns_csr[idx2 + 10] = h_nu_z;
                break;

            case 2:
                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * INV_2DX2 - h_nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * INV_2DX2 + h_nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - h_nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + h_nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(i, j, kp1, 0);
                val_ns_csr[idx2 + 7] = -h_nu_x;

                idx_ns_csr[idx2 + 8] = ijkd2idx(i, j, km1, 0);
                val_ns_csr[idx2 + 8] = h_nu_x;

                idx_ns_csr[idx2 + 9] = ijkd2idx(i, j, kp1, 1);
                val_ns_csr[idx2 + 9] = -h_nu_y;

                idx_ns_csr[idx2 + 10] = ijkd2idx(i, j, km1, 1);
                val_ns_csr[idx2 + 10] = h_nu_y;
                break;
        }

        ptr_ns[idx - is_ns + 1] = idx2 + 11;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 *
                          (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) +
                           u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) +
                           u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)])) *
                          INV_2DX;

        double pressure_term  = -IRHO * calc_gradient_o2_to_o1(pressure, im, d);
        double viscosity_term = 0.5 * eta_s[im] * IRHO * calc_laplacian(u[d], im);

        double vt = 0.;
        switch (d) {
            case 0:
                vt = calc_gradient_o1_to_o1(eta_s, im, 0) * calc_gradient_o1_to_o1(u[0], im, 0) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 1) *
                         (calc_gradient_o1_to_o1(u[0], im, 1) + calc_gradient_o1_to_o1(u[1], im, 0)) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 2) *
                         (calc_gradient_o1_to_o1(u[0], im, 2) + calc_gradient_o1_to_o1(u[2], im, 0));
                break;
            case 1:
                vt = calc_gradient_o1_to_o1(eta_s, im, 1) * calc_gradient_o1_to_o1(u[1], im, 1) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 0) *
                         (calc_gradient_o1_to_o1(u[0], im, 1) + calc_gradient_o1_to_o1(u[1], im, 0)) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 2) *
                         (calc_gradient_o1_to_o1(u[1], im, 2) + calc_gradient_o1_to_o1(u[2], im, 1));
                break;
            case 2:
                vt = calc_gradient_o1_to_o1(eta_s, im, 2) * calc_gradient_o1_to_o1(u[2], im, 2) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 0) *
                         (calc_gradient_o1_to_o1(u[0], im, 2) + calc_gradient_o1_to_o1(u[2], im, 0)) +
                     0.5 * calc_gradient_o1_to_o1(eta_s, im, 1) *
                         (calc_gradient_o1_to_o1(u[1], im, 2) + calc_gradient_o1_to_o1(u[2], im, 1));
                break;
        }
        vt *= IRHO;

        double stress_term = -stress[d][im];
        double bs          = u[d][im] * INV_DT + adv_term + pressure_term + viscosity_term + vt + stress_term;
#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 11, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void NS_MAC_solver_implicit_OBL(double **    u,
                                double *     pressure,
                                double **    u_s,
                                const CTime &jikan,
                                const double degree_oblique,
                                int          is_ns,
                                int          ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const double INV_4DX2 = 1. / (4. * DX2);
    const int    nval     = NX * NY * NZ * DIM;

    const double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;

#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 12 * (idx - is_ns);

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = 1. / jikan.dt_fluid + (3. + gt * gt) * NU / DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - NU * (1. + gt * gt) * INV_2DX2;

        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - NU * (1. + gt * gt) * INV_2DX2;

        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);
        val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 7] = ijkd2idx(ip1, jp1, k, d);
        val_ns_csr[idx2 + 7] = gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 8] = ijkd2idx(ip1, jm1, k, d);
        val_ns_csr[idx2 + 8] = -gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 9] = ijkd2idx(im1, jp1, k, d);
        val_ns_csr[idx2 + 9] = -gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 10] = ijkd2idx(im1, jm1, k, d);
        val_ns_csr[idx2 + 10] = gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 11] = ijkd2idx(i, j, k, 1);
        val_ns_csr[idx2 + 11] = (d == 0) ? Shear_rate_eff : 0.;

        ptr_ns[idx - is_ns + 1] = idx2 + 12;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 * (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) / _2DX +
                                  u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) / _2DX +
                                  u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)]) / _2DX);

        double dp_dx = calc_gradient_o2_to_o1(pressure, im, 0);
        double dp_dy = calc_gradient_o2_to_o1(pressure, im, 1);
        double dp_dz = calc_gradient_o2_to_o1(pressure, im, 2);

        double pressure_term = 0.;  // deriv of scalar -> co to contra
        switch (d) {
            case 0:
                pressure_term = -((1. + gt * gt) * dp_dx - gt * dp_dy);
                break;
            case 1:
                pressure_term = -(-(gt * dp_dx) + dp_dy);
                break;
            case 2:
                pressure_term = -dp_dz;
                break;
        }
        pressure_term *= IRHO;

        double viscosity_term = 0.5 * NU * calc_laplacian_OBL(u[d], im, gt);

        double adv_frame_term = (d == 0) ? -Shear_rate_eff * u[1][im] : 0.;
        double bs             = u[d][im] / jikan.dt_fluid + adv_term + pressure_term + viscosity_term + adv_frame_term;

#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 12, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void CHNS_MAC_solver_implicit_OBL(double **    u,
                                  double *     pressure,
                                  double **    u_s,
                                  double **    stress_s,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ns,
                                  int          ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const double INV_4DX2 = 1. / (4. * DX2);
    const int    nval     = NX * NY * NZ * DIM;

    const double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;

#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 12 * (idx - is_ns);

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = 1. / jikan.dt_fluid + (3. + gt * gt) * NU / DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - NU * (1. + gt * gt) * INV_2DX2;

        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - NU * (1. + gt * gt) * INV_2DX2;

        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);
        val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - NU * INV_2DX2;

        idx_ns_csr[idx2 + 7] = ijkd2idx(ip1, jp1, k, d);
        val_ns_csr[idx2 + 7] = gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 8] = ijkd2idx(ip1, jm1, k, d);
        val_ns_csr[idx2 + 8] = -gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 9] = ijkd2idx(im1, jp1, k, d);
        val_ns_csr[idx2 + 9] = -gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 10] = ijkd2idx(im1, jm1, k, d);
        val_ns_csr[idx2 + 10] = gt * NU * INV_4DX2;

        idx_ns_csr[idx2 + 11] = ijkd2idx(i, j, k, 1);
        val_ns_csr[idx2 + 11] = (d == 0) ? Shear_rate_eff : 0.;

        ptr_ns[idx - is_ns + 1] = idx2 + 12;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 * (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) / _2DX +
                                  u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) / _2DX +
                                  u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)]) / _2DX);

        double dp_dx = calc_gradient_o2_to_o1(pressure, im, 0);
        double dp_dy = calc_gradient_o2_to_o1(pressure, im, 1);
        double dp_dz = calc_gradient_o2_to_o1(pressure, im, 2);

        double pressure_term = 0.;  // deriv of scalar -> co to contra
        switch (d) {
            case 0:
                pressure_term = -((1. + gt * gt) * dp_dx - gt * dp_dy);
                break;
            case 1:
                pressure_term = -(-(gt * dp_dx) + dp_dy);
                break;
            case 2:
                pressure_term = -dp_dz;
                break;
        }
        pressure_term *= IRHO;

        double viscosity_term = 0.5 * NU * calc_laplacian_OBL(u[d], im, gt);
        double stress_term    = -stress[d][im];
        double adv_frame_term = (d == 0) ? -Shear_rate_eff * u[1][im] : 0.;
        double bs =
            u[d][im] / jikan.dt_fluid + adv_term + pressure_term + viscosity_term + adv_frame_term + stress_term;

#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 12, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void CHNS_MAC_solver_implicit_viscosity_OBL(double **    u,
                                            double *     pressure,
                                            double **    u_s,
                                            double **    stress_s,
                                            double *     eta_s,
                                            const CTime &jikan,
                                            const double degree_oblique,
                                            int          is_ns,
                                            int          ie_ns) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2      = DX * DX;
    const double _2DX     = 2. * DX;
    const double INV_DX   = 1. / DX;
    const double INV_2DX  = 1. / _2DX;
    const double INV_DX2  = 1. / DX2;
    const double INV_4DX  = 1. / (4. * DX);
    const double INV_2DX2 = 1. / (2. * DX2);
    const double INV_4DX2 = 1. / (4. * DX2);
    const double INV_DT   = 1. / jikan.dt_fluid;
    const int    nval     = NX * NY * NZ * DIM;

    const double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;

    const double g11 = 1. + gt * gt;
    const double g12 = -gt;
    const double g21 = -gt;
    const double g22 = 1.;
    const double g33 = 1.;

#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im   = ijk2im(i, j, k);
        int idx2 = 20 * (idx - is_ns);

        double nu = eta_s[im] * IRHO;

        double nu_base_x = calc_gradient_o1_to_o1(eta_s, im, 0) * IRHO * INV_2DX;
        double nu_base_y = calc_gradient_o1_to_o1(eta_s, im, 1) * IRHO * INV_2DX;
        double nu_base_z = calc_gradient_o1_to_o1(eta_s, im, 2) * IRHO * INV_2DX;

        double h_nu_x = 0.5 * nu_base_x;
        double h_nu_y = 0.5 * nu_base_y;
        double h_nu_z = 0.5 * nu_base_z;

        idx_ns_csr[idx2 + 0] = ijkd2idx(i, j, k, d);
        val_ns_csr[idx2 + 0] = INV_DT + (3. + gt * gt) * nu * INV_DX2;

        idx_ns_csr[idx2 + 1] = ijkd2idx(ip1, j, k, d);
        idx_ns_csr[idx2 + 2] = ijkd2idx(im1, j, k, d);
        idx_ns_csr[idx2 + 3] = ijkd2idx(i, jp1, k, d);
        idx_ns_csr[idx2 + 4] = ijkd2idx(i, jm1, k, d);
        idx_ns_csr[idx2 + 5] = ijkd2idx(i, j, kp1, d);
        idx_ns_csr[idx2 + 6] = ijkd2idx(i, j, km1, d);

        double nu_x;
        double nu_y;
        double nu_z;
        switch (d) {
            case 0:
                nu_x = g11 * nu_base_x + 0.5 * g21 * nu_base_y;
                nu_y = g12 * nu_base_x + 0.5 * g22 * nu_base_y;
                nu_z = g33 * h_nu_z;

                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 - nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 + nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(ip1, j, k, 1);
                val_ns_csr[idx2 + 7] = -g11 * h_nu_y;

                idx_ns_csr[idx2 + 8] = ijkd2idx(im1, j, k, 1);
                val_ns_csr[idx2 + 8] = g11 * h_nu_y;

                idx_ns_csr[idx2 + 9] = ijkd2idx(i, jp1, k, 1);
                val_ns_csr[idx2 + 9] = -g12 * h_nu_y;

                idx_ns_csr[idx2 + 10] = ijkd2idx(i, jm1, k, 1);
                val_ns_csr[idx2 + 10] = g12 * h_nu_y;

                idx_ns_csr[idx2 + 11] = ijkd2idx(ip1, j, k, 2);
                val_ns_csr[idx2 + 11] = -g11 * h_nu_z;

                idx_ns_csr[idx2 + 12] = ijkd2idx(im1, j, k, 2);
                val_ns_csr[idx2 + 12] = g11 * h_nu_z;

                idx_ns_csr[idx2 + 13] = ijkd2idx(i, jp1, k, 2);
                val_ns_csr[idx2 + 13] = -g12 * h_nu_z;

                idx_ns_csr[idx2 + 14] = ijkd2idx(i, jm1, k, 2);
                val_ns_csr[idx2 + 14] = g12 * h_nu_z;

                break;
            case 1:
                nu_x = 0.5 * g11 * nu_base_x + g21 * nu_base_y;
                nu_y = 0.5 * g12 * nu_base_x + g22 * nu_base_y;
                nu_z = g33 * h_nu_z;

                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 - nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 + nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(ip1, j, k, 0);
                val_ns_csr[idx2 + 7] = -g21 * h_nu_x;

                idx_ns_csr[idx2 + 8] = ijkd2idx(im1, j, k, 0);
                val_ns_csr[idx2 + 8] = g21 * h_nu_x;

                idx_ns_csr[idx2 + 9] = ijkd2idx(i, jp1, k, 0);
                val_ns_csr[idx2 + 9] = -g22 * h_nu_x;

                idx_ns_csr[idx2 + 10] = ijkd2idx(i, jm1, k, 0);
                val_ns_csr[idx2 + 10] = g22 * h_nu_x;

                idx_ns_csr[idx2 + 11] = ijkd2idx(ip1, j, k, 2);
                val_ns_csr[idx2 + 11] = -g21 * h_nu_z;

                idx_ns_csr[idx2 + 12] = ijkd2idx(im1, j, k, 2);
                val_ns_csr[idx2 + 12] = g21 * h_nu_z;

                idx_ns_csr[idx2 + 13] = ijkd2idx(i, jp1, k, 2);
                val_ns_csr[idx2 + 13] = -g22 * h_nu_z;

                idx_ns_csr[idx2 + 14] = ijkd2idx(i, jm1, k, 2);
                val_ns_csr[idx2 + 14] = g22 * h_nu_z;

                break;
            case 2:
                nu_x = 0.5 * g11 * nu_base_x + 0.5 * g21 * nu_base_y;
                nu_y = 0.5 * g12 * nu_base_x + 0.5 * g22 * nu_base_y;
                nu_z = g33 * nu_base_z;

                val_ns_csr[idx2 + 1] = u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 - nu_x;
                val_ns_csr[idx2 + 2] = -u_s[0][im] * INV_4DX - nu * (1. + gt * gt) * INV_2DX2 + nu_x;
                val_ns_csr[idx2 + 3] = u_s[1][im] * INV_4DX - nu * INV_2DX2 - nu_y;
                val_ns_csr[idx2 + 4] = -u_s[1][im] * INV_4DX - nu * INV_2DX2 + nu_y;
                val_ns_csr[idx2 + 5] = u_s[2][im] * INV_4DX - nu * INV_2DX2 - nu_z;
                val_ns_csr[idx2 + 6] = -u_s[2][im] * INV_4DX - nu * INV_2DX2 + nu_z;

                idx_ns_csr[idx2 + 7] = ijkd2idx(i, j, kp1, 0);
                val_ns_csr[idx2 + 7] = -g33 * h_nu_x;

                idx_ns_csr[idx2 + 8] = ijkd2idx(i, j, km1, 0);
                val_ns_csr[idx2 + 8] = g33 * h_nu_x;

                idx_ns_csr[idx2 + 9] = ijkd2idx(i, j, kp1, 1);
                val_ns_csr[idx2 + 9] = -g33 * h_nu_y;

                idx_ns_csr[idx2 + 10] = ijkd2idx(i, j, km1, 1);
                val_ns_csr[idx2 + 10] = g33 * h_nu_y;

                // dummy
                idx_ns_csr[idx2 + 11] = ijkd2idx(i, jp1, k, 0);
                val_ns_csr[idx2 + 11] = 0.;

                idx_ns_csr[idx2 + 12] = ijkd2idx(i, jm1, k, 0);
                val_ns_csr[idx2 + 12] = 0.;

                idx_ns_csr[idx2 + 13] = ijkd2idx(i, jp1, k, 1);
                val_ns_csr[idx2 + 13] = 0.;

                idx_ns_csr[idx2 + 14] = ijkd2idx(i, jm1, k, 1);
                val_ns_csr[idx2 + 14] = 0.;

                break;
        }

        idx_ns_csr[idx2 + 15] = ijkd2idx(ip1, jp1, k, d);
        val_ns_csr[idx2 + 15] = gt * nu * INV_4DX2;

        idx_ns_csr[idx2 + 16] = ijkd2idx(ip1, jm1, k, d);
        val_ns_csr[idx2 + 16] = -gt * nu * INV_4DX2;

        idx_ns_csr[idx2 + 17] = ijkd2idx(im1, jp1, k, d);
        val_ns_csr[idx2 + 17] = -gt * nu * INV_4DX2;

        idx_ns_csr[idx2 + 18] = ijkd2idx(im1, jm1, k, d);
        val_ns_csr[idx2 + 18] = gt * nu * INV_4DX2;

        idx_ns_csr[idx2 + 19] = ijkd2idx(i, j, k, 1);
        val_ns_csr[idx2 + 19] = (d == 0) ? Shear_rate_eff : 0.;

        ptr_ns[idx - is_ns + 1] = idx2 + 20;
    }
    ptr_ns[0] = 0;

    // const vector
#pragma omp parallel for
    for (int idx = is_ns; idx < ie_ns; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        int im = ijk2im(i, j, k);

        double adv_term = -0.5 *
                          (+u_s[0][im] * (u[d][ijk2im(ip1, j, k)] - u[d][ijk2im(im1, j, k)]) +
                           u_s[1][im] * (u[d][ijk2im(i, jp1, k)] - u[d][ijk2im(i, jm1, k)]) +
                           u_s[2][im] * (u[d][ijk2im(i, j, kp1)] - u[d][ijk2im(i, j, km1)])) *
                          INV_2DX;

        double dp_dx = calc_gradient_o2_to_o1(pressure, im, 0);
        double dp_dy = calc_gradient_o2_to_o1(pressure, im, 1);
        double dp_dz = calc_gradient_o2_to_o1(pressure, im, 2);

        double pressure_term = 0.;  // deriv of scalar -> co to contra
        switch (d) {
            case 0:
                pressure_term = -((1. + gt * gt) * dp_dx - gt * dp_dy);
                break;
            case 1:
                pressure_term = -(-(gt * dp_dx) + dp_dy);
                break;
            case 2:
                pressure_term = -dp_dz;
                break;
        }
        pressure_term *= IRHO;

        double viscosity_term = 0.5 * eta_s[im] * IRHO * calc_laplacian_OBL(u[d], im, gt);

        double eta_x = calc_gradient_o1_to_o1(eta_s, im, 0);
        double eta_y = calc_gradient_o1_to_o1(eta_s, im, 1);
        double eta_z = calc_gradient_o1_to_o1(eta_s, im, 2);
        double u_xx  = calc_gradient_o1_to_o1(u[0], im, 0);
        double u_xy  = calc_gradient_o1_to_o1(u[0], im, 1);
        double u_xz  = calc_gradient_o1_to_o1(u[0], im, 2);
        double u_yx  = calc_gradient_o1_to_o1(u[1], im, 0);
        double u_yy  = calc_gradient_o1_to_o1(u[1], im, 1);
        double u_yz  = calc_gradient_o1_to_o1(u[1], im, 2);
        double u_zx  = calc_gradient_o1_to_o1(u[2], im, 0);
        double u_zy  = calc_gradient_o1_to_o1(u[2], im, 1);
        double u_zz  = calc_gradient_o1_to_o1(u[2], im, 2);

        double vt = 0.;
        switch (d) {
            case 0:
                vt = (2. * g11 * eta_x + g21 * eta_y) * u_xx + g11 * eta_y * u_yx + g11 * eta_z * u_zx +
                     (2. * g12 * eta_x + g22 * eta_y) * u_xy + g12 * eta_y * u_yy + g12 * eta_z * u_zy +
                     g33 * eta_z * u_xz;
                break;
            case 1:
                vt = g21 * eta_x * u_xx + (g11 * eta_x + 2. * g21 * eta_y) * u_yx + g21 * eta_z * u_zx +
                     g22 * eta_x * u_xy + (g12 * eta_x + 2. * g22 * eta_y) * u_yy + g22 * eta_z * u_zy +
                     g33 * eta_z * u_yz;
                break;
            case 2:
                vt = (g11 * eta_x + g21 * eta_y) * u_zx + (g12 * eta_x + g22 * eta_y) * u_zy + g33 * eta_x * u_xz +
                     g33 * eta_y * u_yz + 2. * g33 * eta_z * u_zz;
                break;
        }
        vt *= 0.5 * IRHO;

        double stress_term    = -stress[d][im];
        double adv_frame_term = (d == 0) ? -Shear_rate_eff * u[1][im] : 0.;
        double bs = u[d][im] * INV_DT + adv_term + pressure_term + viscosity_term + vt + adv_frame_term + stress_term;

#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ns);
#else
        b_ns[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 20, ptr_ns, idx_ns_csr, val_ns_csr, A_ns);
    lis_matrix_assemble(A_ns);
    lis_solve(A_ns, b_ns, x_ns, lis_solver_ns);
#else
    bicgstab(idx_ns_csr, val_ns_csr, ptr_ns, b_ns, wm_ns, nval);
#endif
    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k, d;
        idx2ijkd(idx, &i, &j, &k, &d);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        u[d][im] = x_ns->value[ijkd2idx(i, j, k, d)];
#else
        u[d][im]  = wm_ns.x[ijkd2idx(i, j, k, d)];
#endif
    }
}

void CH_solver_implicit_euler(double *     psi,
                              double *     psi_o,
                              double *     phi,
                              double **    u,
                              const CTime &jikan,
                              int          is_ch,
                              int          ie_ch) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2     = DX * DX;
    const double _2DX    = 2. * DX;
    const double INV_2DX = 1. / _2DX;
    const double INV_DT  = 1. / jikan.dt_fluid;

    const double aiDX2 = ps.alpha / DX2;
    const double kiDX2 = ps.kappa / DX2;
    const double ak    = aiDX2 * kiDX2;
    const double _2ak  = 2. * ak;
    const int    nval  = NX * NY * NZ;
    static int   call  = 0;

    if (call == 0) {
        call = 1;
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1, jp1, kp1, ip2, jp2, kp2;
            int im1, jm1, km1, im2, jm2, km2;

            ip1 = adj(1, i, NX);
            jp1 = adj(1, j, NY);
            kp1 = adj(1, k, NZ);

            im1 = adj(-1, i, NX);
            jm1 = adj(-1, j, NY);
            km1 = adj(-1, k, NZ);

            ip2 = adj(2, i, NX);
            jp2 = adj(2, j, NY);
            kp2 = adj(2, k, NZ);

            im2 = adj(-2, i, NX);
            jm2 = adj(-2, j, NY);
            km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 25 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] = INV_DT + 42. * ak + 12. * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak;

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak;

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak;

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak;

            idx_ch_csr[idx2 + 5] = ijk2idx(i, j, kp2);
            val_ch_csr[idx2 + 5] = ak;

            idx_ch_csr[idx2 + 6] = ijk2idx(i, j, km2);
            val_ch_csr[idx2 + 6] = ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 7] = ijk2idx(i, jp1, kp1);
            val_ch_csr[idx2 + 7] = _2ak;

            idx_ch_csr[idx2 + 8] = ijk2idx(i, jp1, km1);
            val_ch_csr[idx2 + 8] = _2ak;

            idx_ch_csr[idx2 + 9] = ijk2idx(i, jm1, kp1);
            val_ch_csr[idx2 + 9] = _2ak;

            idx_ch_csr[idx2 + 10] = ijk2idx(i, jm1, km1);
            val_ch_csr[idx2 + 10] = _2ak;

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak;

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak;

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak;

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak;

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak;

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak;

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak;

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] =
                u[0][ijk2im(ip1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] =
                -u[0][ijk2im(im1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //-----------------

            ptr_ch[idx - is_ch + 1] = idx2 + 25;
        }
        ptr_ch[0] = 0;
    } else {
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1 = adj(1, i, NX);
            int jp1 = adj(1, j, NY);
            int kp1 = adj(1, k, NZ);

            int im1 = adj(-1, i, NX);
            int jm1 = adj(-1, j, NY);
            int km1 = adj(-1, k, NZ);

            int ip2 = adj(2, i, NX);
            int jp2 = adj(2, j, NY);
            int kp2 = adj(2, k, NZ);

            int im2 = adj(-2, i, NX);
            int jm2 = adj(-2, j, NY);
            int km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 25 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] = INV_DT + 42. * ak + 12. * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] =
                u[0][ijk2im(ip1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] =
                -u[0][ijk2im(im1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //-----------------
        }
    }

    // const vector
#pragma omp parallel for
    for (int idx = is_ch; idx < ie_ch; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        double psi_im = psi[ijk2im(i, j, k)];
        double bs;
        bs = psi_im * INV_DT +
                kiDX2 * (potential_deriv(psi[ijk2im(ip1, j, k)]) + potential_deriv(psi[ijk2im(im1, j, k)]) +
                         potential_deriv(psi[ijk2im(i, jp1, k)]) + potential_deriv(psi[ijk2im(i, jm1, k)]) +
                         potential_deriv(psi[ijk2im(i, j, kp1)]) + potential_deriv(psi[ijk2im(i, j, km1)]) -
                        6. * potential_deriv(psi_im)) -
                2. * ps.neutral * ps.d * kiDX2 *
                    (phi[ijk2im(ip1, j, k)] + phi[ijk2im(im1, j, k)] + phi[ijk2im(i, jp1, k)] + phi[ijk2im(i, jm1, k)] +
                     phi[ijk2im(i, j, kp1)] + phi[ijk2im(i, j, km1)] - 6. * phi[ijk2im(i, j, k)]);

        if (SW_WALL != NO_WALL){
            bs += ps.w * A_XI * kiDX2 *
                    (calc_gradient_norm(phi_p, ijk2im(ip1, j, k)) + calc_gradient_norm(phi_p, ijk2im(im1, j, k)) +
                     calc_gradient_norm(phi_p, ijk2im(i, jp1, k)) + calc_gradient_norm(phi_p, ijk2im(i, jm1, k)) +
                     calc_gradient_norm(phi_p, ijk2im(i, j, kp1)) + calc_gradient_norm(phi_p, ijk2im(i, j, km1)) -
                     6. * calc_gradient_norm(phi_p, ijk2im(i, j, k)));
            if (k <= 0.5 * NZ){
                bs += ps.w_wall * A_XI * kiDX2 *
                      (calc_gradient_norm(phi_wall, ijk2im(ip1, j, k)) + calc_gradient_norm(phi_wall, ijk2im(im1, j, k)) +
                       calc_gradient_norm(phi_wall, ijk2im(i, jp1, k)) + calc_gradient_norm(phi_wall, ijk2im(i, jm1, k)) +
                       calc_gradient_norm(phi_wall, ijk2im(i, j, kp1)) + calc_gradient_norm(phi_wall, ijk2im(i, j, km1)) -
                       6. * calc_gradient_norm(phi_wall, ijk2im(i, j, k)));
            }
        } else {
            bs += ps.w * A_XI * kiDX2 *
                    (calc_gradient_norm(phi, ijk2im(ip1, j, k)) + calc_gradient_norm(phi, ijk2im(im1, j, k)) +
                     calc_gradient_norm(phi, ijk2im(i, jp1, k)) + calc_gradient_norm(phi, ijk2im(i, jm1, k)) +
                     calc_gradient_norm(phi, ijk2im(i, j, kp1)) + calc_gradient_norm(phi, ijk2im(i, j, km1)) -
                     6. * calc_gradient_norm(phi, ijk2im(i, j, k)));
        }
#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ch);
#else
        b_ch[idx] = bs;
#endif
    }
    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 25, ptr_ch, idx_ch_csr, val_ch_csr, A_ch);
    lis_matrix_assemble(A_ch);
    lis_solve(A_ch, b_ch, x_ch, lis_solver_ch);
#else
    bicgstab(idx_ch_csr, val_ch_csr, ptr_ch, b_ch, wm_ch, nval);
#endif

    Cpy_v1(psi_o, psi);

    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        psi[im] = x_ch->value[idx];
#else
        psi[im]   = wm_ch.x[ijk2idx(i, j, k)];
#endif
    }
}

void CH_solver_implicit_bdfab(double *     psi,
                              double *     psi_o,
                              double *     phi,
                              double **    u,
                              const CTime &jikan,
                              int          is_ch,
                              int          ie_ch) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2     = DX * DX;
    const double _2DX    = 2. * DX;
    const double INV_2DX = 1. / _2DX;
    const double INV_DT  = 1. / jikan.dt_fluid;

    const double aiDX2 = ps.alpha / DX2;
    const double kiDX2 = ps.kappa / DX2;
    const double ak    = aiDX2 * kiDX2;
    const double _2ak  = 2. * ak;
    const int    nval  = NX * NY * NZ;
    static int   call  = 0;

    if (call == 0) {
        call = 1;
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1, jp1, kp1, ip2, jp2, kp2;
            int im1, jm1, km1, im2, jm2, km2;

            ip1 = adj(1, i, NX);
            jp1 = adj(1, j, NY);
            kp1 = adj(1, k, NZ);

            im1 = adj(-1, i, NX);
            jm1 = adj(-1, j, NY);
            km1 = adj(-1, k, NZ);

            ip2 = adj(2, i, NX);
            jp2 = adj(2, j, NY);
            kp2 = adj(2, k, NZ);

            im2 = adj(-2, i, NX);
            jm2 = adj(-2, j, NY);
            km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 25 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] = 1.5 * INV_DT + 42. * ak + 12. * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak;

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak;

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak;

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak;

            idx_ch_csr[idx2 + 5] = ijk2idx(i, j, kp2);
            val_ch_csr[idx2 + 5] = ak;

            idx_ch_csr[idx2 + 6] = ijk2idx(i, j, km2);
            val_ch_csr[idx2 + 6] = ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 7] = ijk2idx(i, jp1, kp1);
            val_ch_csr[idx2 + 7] = _2ak;

            idx_ch_csr[idx2 + 8] = ijk2idx(i, jp1, km1);
            val_ch_csr[idx2 + 8] = _2ak;

            idx_ch_csr[idx2 + 9] = ijk2idx(i, jm1, kp1);
            val_ch_csr[idx2 + 9] = _2ak;

            idx_ch_csr[idx2 + 10] = ijk2idx(i, jm1, km1);
            val_ch_csr[idx2 + 10] = _2ak;

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak;

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak;

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak;

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak;

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak;

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak;

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak;

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] =
                u[0][ijk2im(ip1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] =
                -u[0][ijk2im(im1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //-----------------

            ptr_ch[idx - is_ch + 1] = idx2 + 25;
        }
        ptr_ch[0] = 0;
    } else {
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1 = adj(1, i, NX);
            int jp1 = adj(1, j, NY);
            int kp1 = adj(1, k, NZ);

            int im1 = adj(-1, i, NX);
            int jm1 = adj(-1, j, NY);
            int km1 = adj(-1, k, NZ);

            int ip2 = adj(2, i, NX);
            int jp2 = adj(2, j, NY);
            int kp2 = adj(2, k, NZ);

            int im2 = adj(-2, i, NX);
            int jm2 = adj(-2, j, NY);
            int km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 25 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] = 1.5 * INV_DT + 42. * ak + 12. * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] =
                u[0][ijk2im(ip1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] =
                -u[0][ijk2im(im1, j, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 12. * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //-----------------
        }
    }

    // const vector
#pragma omp parallel for
    for (int idx = is_ch; idx < ie_ch; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        double psi_im   = psi[ijk2im(i, j, k)];
        double psi_o_im = psi_o[ijk2im(i, j, k)];
        double bs;

        bs =0.5 * (4. * psi_im - psi_o_im) * INV_DT +
                2. * kiDX2 *
                    (potential_deriv(psi[ijk2im(ip1, j, k)]) + potential_deriv(psi[ijk2im(im1, j, k)]) +
                     potential_deriv(psi[ijk2im(i, jp1, k)]) + potential_deriv(psi[ijk2im(i, jm1, k)]) +
                     potential_deriv(psi[ijk2im(i, j, kp1)]) + potential_deriv(psi[ijk2im(i, j, km1)]) -
                     6. * potential_deriv(psi_im)) -
                kiDX2 * (potential_deriv(psi_o[ijk2im(ip1, j, k)]) + potential_deriv(psi_o[ijk2im(im1, j, k)]) +
                         potential_deriv(psi_o[ijk2im(i, jp1, k)]) + potential_deriv(psi_o[ijk2im(i, jm1, k)]) +
                         potential_deriv(psi_o[ijk2im(i, j, kp1)]) + potential_deriv(psi_o[ijk2im(i, j, km1)]) -
                         6. * potential_deriv(psi_o_im)) -        
                2. * ps.neutral * ps.d * kiDX2 *
                    (phi[ijk2im(ip1, j, k)] + phi[ijk2im(im1, j, k)] + phi[ijk2im(i, jp1, k)] + phi[ijk2im(i, jm1, k)] +
                     phi[ijk2im(i, j, kp1)] + phi[ijk2im(i, j, km1)] - 6. * phi[ijk2im(i, j, k)]);
        if (SW_WALL != NO_WALL){
            bs +=ps.w * A_XI * kiDX2 *
                    (calc_gradient_norm(phi_p, ijk2im(ip1, j, k)) + calc_gradient_norm(phi_p, ijk2im(im1, j, k)) +
                     calc_gradient_norm(phi_p, ijk2im(i, jp1, k)) + calc_gradient_norm(phi_p, ijk2im(i, jm1, k)) +
                     calc_gradient_norm(phi_p, ijk2im(i, j, kp1)) + calc_gradient_norm(phi_p, ijk2im(i, j, km1)) -
                     6. * calc_gradient_norm(phi_p, ijk2im(i, j, k)));
            if (k <= 0.5 * NZ){
                bs += ps.w_wall * A_XI * kiDX2 *
                      (calc_gradient_norm(phi_wall, ijk2im(ip1, j, k)) + calc_gradient_norm(phi_wall, ijk2im(im1, j, k)) +
                       calc_gradient_norm(phi_wall, ijk2im(i, jp1, k)) + calc_gradient_norm(phi_wall, ijk2im(i, jm1, k)) +
                       calc_gradient_norm(phi_wall, ijk2im(i, j, kp1)) + calc_gradient_norm(phi_wall, ijk2im(i, j, km1)) -
                       6. * calc_gradient_norm(phi_wall, ijk2im(i, j, k)));
            }
                
        } else {
            bs +=ps.w * A_XI * kiDX2 *
                    (calc_gradient_norm(phi, ijk2im(ip1, j, k)) + calc_gradient_norm(phi, ijk2im(im1, j, k)) +
                     calc_gradient_norm(phi, ijk2im(i, jp1, k)) + calc_gradient_norm(phi, ijk2im(i, jm1, k)) +
                     calc_gradient_norm(phi, ijk2im(i, j, kp1)) + calc_gradient_norm(phi, ijk2im(i, j, km1)) -
                     6. * calc_gradient_norm(phi, ijk2im(i, j, k)));
        }
#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ch);
#else
        b_ch[idx] = bs;
#endif
    }

    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 25, ptr_ch, idx_ch_csr, val_ch_csr, A_ch);
    lis_matrix_assemble(A_ch);
    lis_solve(A_ch, b_ch, x_ch, lis_solver_ch);
#else
    bicgstab(idx_ch_csr, val_ch_csr, ptr_ch, b_ch, wm_ch, nval);
#endif

    Cpy_v1(psi_o, psi);

    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        psi[im] = x_ch->value[idx];
#else
        psi[im]   = wm_ch.x[ijk2idx(i, j, k)];
#endif
    }
}

void CH_solver_implicit_euler_OBL(double *     psi,
                                  double *     psi_o,
                                  double *     phi,
                                  double **    u,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ch,
                                  int          ie_ch) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2     = DX * DX;
    const double _2DX    = 2. * DX;
    const double INV_2DX = 1. / _2DX;
    const double INV_DT  = 1. / jikan.dt_fluid;

    const double aiDX2 = ps.alpha / DX2;
    const double kiDX2 = ps.kappa / DX2;
    const double ak    = aiDX2 * kiDX2;
    const double _2ak  = 2. * ak;

    const double gt  = degree_oblique;
    const double gt2 = gt * gt;
    const double gt3 = gt2 * gt;
    const double gt4 = gt2 * gt2;

    const int  nval = NX * NY * NZ;
    static int call = 0;

    if (call == 0) {
        call = 1;
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1, jp1, kp1, ip2, jp2, kp2;
            int im1, jm1, km1, im2, jm2, km2;

            ip1 = adj(1, i, NX);
            jp1 = adj(1, j, NY);
            kp1 = adj(1, k, NZ);

            im1 = adj(-1, i, NX);
            jm1 = adj(-1, j, NY);
            km1 = adj(-1, k, NZ);

            ip2 = adj(2, i, NX);
            jp2 = adj(2, j, NY);
            kp2 = adj(2, k, NZ);

            im2 = adj(-2, i, NX);
            jm2 = adj(-2, j, NY);
            km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 45 * (idx - is_ch);

            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] =
                INV_DT + (42. + 29. * gt2 + 6. * gt4) * ak + 4. * (3. + gt2) * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 5] = ijk2idx(i, j, kp2);
            val_ch_csr[idx2 + 5] = ak;

            idx_ch_csr[idx2 + 6] = ijk2idx(i, j, km2);
            val_ch_csr[idx2 + 6] = ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 7] = ijk2idx(i, jp1, kp1);
            val_ch_csr[idx2 + 7] = _2ak;

            idx_ch_csr[idx2 + 8] = ijk2idx(i, jp1, km1);
            val_ch_csr[idx2 + 8] = _2ak;

            idx_ch_csr[idx2 + 9] = ijk2idx(i, jm1, kp1);
            val_ch_csr[idx2 + 9] = _2ak;

            idx_ch_csr[idx2 + 10] = ijk2idx(i, jm1, km1);
            val_ch_csr[idx2 + 10] = _2ak;

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak * (1. + gt2);

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(ip1, jp1, k)];

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(ip1, jm1, k)];

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(im1, jp1, k)];

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(im1, jm1, k)];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] = u[0][ijk2im(ip1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] = -u[0][ijk2im(im1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //---------------------------------------

            idx_ch_csr[idx2 + 25] = ijk2idx(ip1, jp1, kp1);
            val_ch_csr[idx2 + 25] = -ak * gt;

            idx_ch_csr[idx2 + 26] = ijk2idx(ip1, jp1, km1);
            val_ch_csr[idx2 + 26] = -ak * gt;

            idx_ch_csr[idx2 + 27] = ijk2idx(im1, jm1, kp1);
            val_ch_csr[idx2 + 27] = -ak * gt;

            idx_ch_csr[idx2 + 28] = ijk2idx(im1, jm1, km1);
            val_ch_csr[idx2 + 28] = -ak * gt;

            idx_ch_csr[idx2 + 29] = ijk2idx(ip1, jp2, k);
            val_ch_csr[idx2 + 29] = -ak * gt;

            idx_ch_csr[idx2 + 30] = ijk2idx(im1, jm2, k);
            val_ch_csr[idx2 + 30] = -ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 31] = ijk2idx(ip1, jm1, kp1);
            val_ch_csr[idx2 + 31] = ak * gt;

            idx_ch_csr[idx2 + 32] = ijk2idx(ip1, jm1, km1);
            val_ch_csr[idx2 + 32] = ak * gt;

            idx_ch_csr[idx2 + 33] = ijk2idx(im1, jp1, kp1);
            val_ch_csr[idx2 + 33] = ak * gt;

            idx_ch_csr[idx2 + 34] = ijk2idx(im1, jp1, km1);
            val_ch_csr[idx2 + 34] = ak * gt;

            idx_ch_csr[idx2 + 35] = ijk2idx(ip1, jm2, k);
            val_ch_csr[idx2 + 35] = ak * gt;

            idx_ch_csr[idx2 + 36] = ijk2idx(im1, jp2, k);
            val_ch_csr[idx2 + 36] = ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 37] = ijk2idx(ip2, jp1, k);
            val_ch_csr[idx2 + 37] = -ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 38] = ijk2idx(im2, jm1, k);
            val_ch_csr[idx2 + 38] = -ak * gt * (1. + gt2);

            //-----------------

            idx_ch_csr[idx2 + 39] = ijk2idx(ip2, jm1, k);
            val_ch_csr[idx2 + 39] = ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 40] = ijk2idx(im2, jp1, k);
            val_ch_csr[idx2 + 40] = ak * gt * (1. + gt2);

            //---------------------------------------

            idx_ch_csr[idx2 + 41] = ijk2idx(ip2, jp2, k);
            val_ch_csr[idx2 + 41] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 42] = ijk2idx(im2, jm2, k);
            val_ch_csr[idx2 + 42] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 43] = ijk2idx(ip2, jm2, k);
            val_ch_csr[idx2 + 43] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 44] = ijk2idx(im2, jp2, k);
            val_ch_csr[idx2 + 44] = ak * gt2 * 0.25;

            ptr_ch[idx - is_ch + 1] = idx2 + 45;
        }
        ptr_ch[0] = 0;
    } else {
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1 = adj(1, i, NX);
            int jp1 = adj(1, j, NY);
            int kp1 = adj(1, k, NZ);

            int im1 = adj(-1, i, NX);
            int jm1 = adj(-1, j, NY);
            int km1 = adj(-1, k, NZ);

            int ip2 = adj(2, i, NX);
            int jp2 = adj(2, j, NY);
            int kp2 = adj(2, k, NZ);

            int im2 = adj(-2, i, NX);
            int jm2 = adj(-2, j, NY);
            int km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 45 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] =
                INV_DT + (42. + 29. * gt2 + 6. * gt4) * ak + 4. * (3. + gt2) * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak * (1. - 0.5 * gt2);

            //---------------------------------------

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak * (1. + gt2);

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(ip1, jp1, k)];

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(ip1, jm1, k)];

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(im1, jp1, k)];

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(im1, jm1, k)];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] = u[0][ijk2im(ip1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] = -u[0][ijk2im(im1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //---------------------------------------

            idx_ch_csr[idx2 + 25] = ijk2idx(ip1, jp1, kp1);
            val_ch_csr[idx2 + 25] = -ak * gt;

            idx_ch_csr[idx2 + 26] = ijk2idx(ip1, jp1, km1);
            val_ch_csr[idx2 + 26] = -ak * gt;

            idx_ch_csr[idx2 + 27] = ijk2idx(im1, jm1, kp1);
            val_ch_csr[idx2 + 27] = -ak * gt;

            idx_ch_csr[idx2 + 28] = ijk2idx(im1, jm1, km1);
            val_ch_csr[idx2 + 28] = -ak * gt;

            idx_ch_csr[idx2 + 29] = ijk2idx(ip1, jp2, k);
            val_ch_csr[idx2 + 29] = -ak * gt;

            idx_ch_csr[idx2 + 30] = ijk2idx(im1, jm2, k);
            val_ch_csr[idx2 + 30] = -ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 31] = ijk2idx(ip1, jm1, kp1);
            val_ch_csr[idx2 + 31] = ak * gt;

            idx_ch_csr[idx2 + 32] = ijk2idx(ip1, jm1, km1);
            val_ch_csr[idx2 + 32] = ak * gt;

            idx_ch_csr[idx2 + 33] = ijk2idx(im1, jp1, kp1);
            val_ch_csr[idx2 + 33] = ak * gt;

            idx_ch_csr[idx2 + 34] = ijk2idx(im1, jp1, km1);
            val_ch_csr[idx2 + 34] = ak * gt;

            idx_ch_csr[idx2 + 35] = ijk2idx(ip1, jm2, k);
            val_ch_csr[idx2 + 35] = ak * gt;

            idx_ch_csr[idx2 + 36] = ijk2idx(im1, jp2, k);
            val_ch_csr[idx2 + 36] = ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 37] = ijk2idx(ip2, jp1, k);
            val_ch_csr[idx2 + 37] = -ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 38] = ijk2idx(im2, jm1, k);
            val_ch_csr[idx2 + 38] = -ak * gt * (1. + gt2);

            //-----------------

            idx_ch_csr[idx2 + 39] = ijk2idx(ip2, jm1, k);
            val_ch_csr[idx2 + 39] = ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 40] = ijk2idx(im2, jp1, k);
            val_ch_csr[idx2 + 40] = ak * gt * (1. + gt2);

            //---------------------------------------

            idx_ch_csr[idx2 + 41] = ijk2idx(ip2, jp2, k);
            val_ch_csr[idx2 + 41] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 42] = ijk2idx(im2, jm2, k);
            val_ch_csr[idx2 + 42] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 43] = ijk2idx(ip2, jm2, k);
            val_ch_csr[idx2 + 43] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 44] = ijk2idx(im2, jp2, k);
            val_ch_csr[idx2 + 44] = ak * gt2 * 0.25;
        }
    }

    // const vector
#pragma omp parallel for
    for (int idx = is_ch; idx < ie_ch; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        double psi_im = psi[ijk2im(i, j, k)];

        double bs =
            psi_im * INV_DT +
            kiDX2 * ((1. + gt2) * (potential_deriv(psi[ijk2im(ip1, j, k)]) + potential_deriv(psi[ijk2im(im1, j, k)])) +
                     potential_deriv(psi[ijk2im(i, jp1, k)]) + potential_deriv(psi[ijk2im(i, jm1, k)]) +
                     potential_deriv(psi[ijk2im(i, j, kp1)]) + potential_deriv(psi[ijk2im(i, j, km1)]) -
                     0.5 * gt *
                         (potential_deriv(psi[ijk2im(ip1, jp1, k)]) - potential_deriv(psi[ijk2im(ip1, jm1, k)]) -
                          potential_deriv(psi[ijk2im(im1, jp1, k)]) + potential_deriv(psi[ijk2im(im1, jm1, k)])) -
                     2. * (3. + gt2) * potential_deriv(psi_im)) +
            ps.w * A_XI * kiDX2 *
                ((1. + gt2) * (calc_gradient_norm_OBL(phi, ijk2im(ip1, j, k), gt) +
                               calc_gradient_norm_OBL(phi, ijk2im(im1, j, k), gt)) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, jp1, k), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, jm1, k), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, j, kp1), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, j, km1), gt) -
                 0.5 * gt *
                     (calc_gradient_norm_OBL(phi, ijk2im(ip1, jp1, k), gt) -
                      calc_gradient_norm_OBL(phi, ijk2im(ip1, jm1, k), gt) -
                      calc_gradient_norm_OBL(phi, ijk2im(im1, jp1, k), gt) +
                      calc_gradient_norm_OBL(phi, ijk2im(im1, jm1, k), gt)) -
                 2. * (3. + gt2) * calc_gradient_norm_OBL(phi, ijk2im(i, j, k), gt)) -
            2. * ps.neutral * ps.d * kiDX2 *
                ((1. + gt2) * (phi[ijk2im(ip1, j, k)] + phi[ijk2im(im1, j, k)]) + phi[ijk2im(i, jp1, k)] +
                 phi[ijk2im(i, jm1, k)] + phi[ijk2im(i, j, kp1)] + phi[ijk2im(i, j, km1)] -
                 0.5 * gt *
                     (phi[ijk2im(ip1, jp1, k)] - phi[ijk2im(ip1, jm1, k)] - phi[ijk2im(im1, jp1, k)] +
                      phi[ijk2im(im1, jm1, k)]) -
                 2. * (3. + gt2) * phi[ijk2im(i, j, k)]);

#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ch);
#else
        b_ch[idx] = bs;
#endif
    }
    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 45, ptr_ch, idx_ch_csr, val_ch_csr, A_ch);
    lis_matrix_assemble(A_ch);
    lis_solve(A_ch, b_ch, x_ch, lis_solver_ch);
    // if (jikan.ts % 10 == 0) {
    //	lis_solver_get_iter(lis_solver_ch, &iter);
    //	printf("%d CH matrix solver iter = %d\n", jikan.ts, iter);
    //}
#else
    bicgstab(idx_ch_csr, val_ch_csr, ptr_ch, b_ch, wm_ch, nval);
#endif

    Cpy_v1(psi_o, psi);

    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        psi[im] = x_ch->value[idx];
#else
        psi[im]   = wm_ch.x[ijk2idx(i, j, k)];
#endif
    }
}

void CH_solver_implicit_bdfab_OBL(double *     psi,
                                  double *     psi_o,
                                  double *     phi,
                                  double **    u,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ch,
                                  int          ie_ch) {
#ifdef _LIS_SOLVER
    LIS_INT iter;
#endif
    const double DX2     = DX * DX;
    const double _2DX    = 2. * DX;
    const double INV_2DX = 1. / _2DX;
    const double INV_DT  = 1. / jikan.dt_fluid;

    const double aiDX2 = ps.alpha / DX2;
    const double kiDX2 = ps.kappa / DX2;
    const double ak    = aiDX2 * kiDX2;
    const double _2ak  = 2. * ak;

    const double gt  = degree_oblique;
    const double gt2 = gt * gt;
    const double gt3 = gt2 * gt;
    const double gt4 = gt2 * gt2;

    const int  nval = NX * NY * NZ;
    static int call = 0;

    if (call == 0) {
        call = 1;
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1, jp1, kp1, ip2, jp2, kp2;
            int im1, jm1, km1, im2, jm2, km2;

            ip1 = adj(1, i, NX);
            jp1 = adj(1, j, NY);
            kp1 = adj(1, k, NZ);

            im1 = adj(-1, i, NX);
            jm1 = adj(-1, j, NY);
            km1 = adj(-1, k, NZ);

            ip2 = adj(2, i, NX);
            jp2 = adj(2, j, NY);
            kp2 = adj(2, k, NZ);

            im2 = adj(-2, i, NX);
            jm2 = adj(-2, j, NY);
            km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 45 * (idx - is_ch);

            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] =
                1.5 * INV_DT + (42. + 29. * gt2 + 6. * gt4) * ak + 4. * (3. + gt2) * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 5] = ijk2idx(i, j, kp2);
            val_ch_csr[idx2 + 5] = ak;

            idx_ch_csr[idx2 + 6] = ijk2idx(i, j, km2);
            val_ch_csr[idx2 + 6] = ak;

            //---------------------------------------

            idx_ch_csr[idx2 + 7] = ijk2idx(i, jp1, kp1);
            val_ch_csr[idx2 + 7] = _2ak;

            idx_ch_csr[idx2 + 8] = ijk2idx(i, jp1, km1);
            val_ch_csr[idx2 + 8] = _2ak;

            idx_ch_csr[idx2 + 9] = ijk2idx(i, jm1, kp1);
            val_ch_csr[idx2 + 9] = _2ak;

            idx_ch_csr[idx2 + 10] = ijk2idx(i, jm1, km1);
            val_ch_csr[idx2 + 10] = _2ak;

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak * (1. + gt2);

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(ip1, jp1, k)];

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(ip1, jm1, k)];

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(im1, jp1, k)];

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(im1, jm1, k)];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] = u[0][ijk2im(ip1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] = -u[0][ijk2im(im1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //---------------------------------------

            idx_ch_csr[idx2 + 25] = ijk2idx(ip1, jp1, kp1);
            val_ch_csr[idx2 + 25] = -ak * gt;

            idx_ch_csr[idx2 + 26] = ijk2idx(ip1, jp1, km1);
            val_ch_csr[idx2 + 26] = -ak * gt;

            idx_ch_csr[idx2 + 27] = ijk2idx(im1, jm1, kp1);
            val_ch_csr[idx2 + 27] = -ak * gt;

            idx_ch_csr[idx2 + 28] = ijk2idx(im1, jm1, km1);
            val_ch_csr[idx2 + 28] = -ak * gt;

            idx_ch_csr[idx2 + 29] = ijk2idx(ip1, jp2, k);
            val_ch_csr[idx2 + 29] = -ak * gt;

            idx_ch_csr[idx2 + 30] = ijk2idx(im1, jm2, k);
            val_ch_csr[idx2 + 30] = -ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 31] = ijk2idx(ip1, jm1, kp1);
            val_ch_csr[idx2 + 31] = ak * gt;

            idx_ch_csr[idx2 + 32] = ijk2idx(ip1, jm1, km1);
            val_ch_csr[idx2 + 32] = ak * gt;

            idx_ch_csr[idx2 + 33] = ijk2idx(im1, jp1, kp1);
            val_ch_csr[idx2 + 33] = ak * gt;

            idx_ch_csr[idx2 + 34] = ijk2idx(im1, jp1, km1);
            val_ch_csr[idx2 + 34] = ak * gt;

            idx_ch_csr[idx2 + 35] = ijk2idx(ip1, jm2, k);
            val_ch_csr[idx2 + 35] = ak * gt;

            idx_ch_csr[idx2 + 36] = ijk2idx(im1, jp2, k);
            val_ch_csr[idx2 + 36] = ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 37] = ijk2idx(ip2, jp1, k);
            val_ch_csr[idx2 + 37] = -ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 38] = ijk2idx(im2, jm1, k);
            val_ch_csr[idx2 + 38] = -ak * gt * (1. + gt2);

            //-----------------

            idx_ch_csr[idx2 + 39] = ijk2idx(ip2, jm1, k);
            val_ch_csr[idx2 + 39] = ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 40] = ijk2idx(im2, jp1, k);
            val_ch_csr[idx2 + 40] = ak * gt * (1. + gt2);

            //---------------------------------------

            idx_ch_csr[idx2 + 41] = ijk2idx(ip2, jp2, k);
            val_ch_csr[idx2 + 41] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 42] = ijk2idx(im2, jm2, k);
            val_ch_csr[idx2 + 42] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 43] = ijk2idx(ip2, jm2, k);
            val_ch_csr[idx2 + 43] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 44] = ijk2idx(im2, jp2, k);
            val_ch_csr[idx2 + 44] = ak * gt2 * 0.25;

            ptr_ch[idx - is_ch + 1] = idx2 + 45;
        }
        ptr_ch[0] = 0;
    } else {
#pragma omp parallel for
        for (int idx = is_ch; idx < ie_ch; idx++) {
            int i, j, k;
            idx2ijk(idx, &i, &j, &k);

            int ip1 = adj(1, i, NX);
            int jp1 = adj(1, j, NY);
            int kp1 = adj(1, k, NZ);

            int im1 = adj(-1, i, NX);
            int jm1 = adj(-1, j, NY);
            int km1 = adj(-1, k, NZ);

            int ip2 = adj(2, i, NX);
            int jp2 = adj(2, j, NY);
            int kp2 = adj(2, k, NZ);

            int im2 = adj(-2, i, NX);
            int jm2 = adj(-2, j, NY);
            int km2 = adj(-2, k, NZ);

            int im = ijk2im(i, j, k);

            int idx2 = 45 * (idx - is_ch);
            //---------------------------------------

            idx_ch_csr[idx2 + 0] = ijk2idx(i, j, k);
            val_ch_csr[idx2 + 0] =
                1.5 * INV_DT + (42. + 29. * gt2 + 6. * gt4) * ak + 4. * (3. + gt2) * kiDX2 * ps.d * phi[im];

            //---------------------------------------

            idx_ch_csr[idx2 + 1] = ijk2idx(ip2, j, k);
            val_ch_csr[idx2 + 1] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 2] = ijk2idx(im2, j, k);
            val_ch_csr[idx2 + 2] = ak * (1. + 1.5 * gt2 + gt4);

            idx_ch_csr[idx2 + 3] = ijk2idx(i, jp2, k);
            val_ch_csr[idx2 + 3] = ak * (1. - 0.5 * gt2);

            idx_ch_csr[idx2 + 4] = ijk2idx(i, jm2, k);
            val_ch_csr[idx2 + 4] = ak * (1. - 0.5 * gt2);

            //---------------------------------------

            //-----------------

            idx_ch_csr[idx2 + 11] = ijk2idx(ip1, j, kp1);
            val_ch_csr[idx2 + 11] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 12] = ijk2idx(ip1, j, km1);
            val_ch_csr[idx2 + 12] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 13] = ijk2idx(im1, j, kp1);
            val_ch_csr[idx2 + 13] = _2ak * (1. + gt2);

            idx_ch_csr[idx2 + 14] = ijk2idx(im1, j, km1);
            val_ch_csr[idx2 + 14] = _2ak * (1. + gt2);

            //-----------------
            idx_ch_csr[idx2 + 15] = ijk2idx(ip1, jp1, k);
            val_ch_csr[idx2 + 15] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(ip1, jp1, k)];

            idx_ch_csr[idx2 + 16] = ijk2idx(ip1, jm1, k);
            val_ch_csr[idx2 + 16] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(ip1, jm1, k)];

            idx_ch_csr[idx2 + 17] = ijk2idx(im1, jp1, k);
            val_ch_csr[idx2 + 17] = _2ak * (1. - 3. * gt + gt2 - gt3) - ps.d * kiDX2 * gt * phi[ijk2im(im1, jp1, k)];

            idx_ch_csr[idx2 + 18] = ijk2idx(im1, jm1, k);
            val_ch_csr[idx2 + 18] = _2ak * (1. + 3. * gt + gt2 + gt3) + ps.d * kiDX2 * gt * phi[ijk2im(im1, jm1, k)];

            //---------------------------------------

            idx_ch_csr[idx2 + 19] = ijk2idx(ip1, j, k);
            val_ch_csr[idx2 + 19] = u[0][ijk2im(ip1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(ip1, j, k)];

            idx_ch_csr[idx2 + 20] = ijk2idx(im1, j, k);
            val_ch_csr[idx2 + 20] = -u[0][ijk2im(im1, j, k)] * INV_2DX - 4. * (3. + 4. * gt2 + gt4) * ak -
                                    (1. + gt2) * 2. * kiDX2 * ps.d * phi[ijk2im(im1, j, k)];

            //-----------------

            idx_ch_csr[idx2 + 21] = ijk2idx(i, jp1, k);
            val_ch_csr[idx2 + 21] =
                u[1][ijk2im(i, jp1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jp1, k)];

            idx_ch_csr[idx2 + 22] = ijk2idx(i, jm1, k);
            val_ch_csr[idx2 + 22] =
                -u[1][ijk2im(i, jm1, k)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, jm1, k)];

            //-----------------

            idx_ch_csr[idx2 + 23] = ijk2idx(i, j, kp1);
            val_ch_csr[idx2 + 23] =
                u[2][ijk2im(i, j, kp1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, kp1)];

            idx_ch_csr[idx2 + 24] = ijk2idx(i, j, km1);
            val_ch_csr[idx2 + 24] =
                -u[2][ijk2im(i, j, km1)] * INV_2DX - 4. * (3. + gt2) * ak - 2. * kiDX2 * ps.d * phi[ijk2im(i, j, km1)];

            //---------------------------------------

            idx_ch_csr[idx2 + 25] = ijk2idx(ip1, jp1, kp1);
            val_ch_csr[idx2 + 25] = -ak * gt;

            idx_ch_csr[idx2 + 26] = ijk2idx(ip1, jp1, km1);
            val_ch_csr[idx2 + 26] = -ak * gt;

            idx_ch_csr[idx2 + 27] = ijk2idx(im1, jm1, kp1);
            val_ch_csr[idx2 + 27] = -ak * gt;

            idx_ch_csr[idx2 + 28] = ijk2idx(im1, jm1, km1);
            val_ch_csr[idx2 + 28] = -ak * gt;

            idx_ch_csr[idx2 + 29] = ijk2idx(ip1, jp2, k);
            val_ch_csr[idx2 + 29] = -ak * gt;

            idx_ch_csr[idx2 + 30] = ijk2idx(im1, jm2, k);
            val_ch_csr[idx2 + 30] = -ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 31] = ijk2idx(ip1, jm1, kp1);
            val_ch_csr[idx2 + 31] = ak * gt;

            idx_ch_csr[idx2 + 32] = ijk2idx(ip1, jm1, km1);
            val_ch_csr[idx2 + 32] = ak * gt;

            idx_ch_csr[idx2 + 33] = ijk2idx(im1, jp1, kp1);
            val_ch_csr[idx2 + 33] = ak * gt;

            idx_ch_csr[idx2 + 34] = ijk2idx(im1, jp1, km1);
            val_ch_csr[idx2 + 34] = ak * gt;

            idx_ch_csr[idx2 + 35] = ijk2idx(ip1, jm2, k);
            val_ch_csr[idx2 + 35] = ak * gt;

            idx_ch_csr[idx2 + 36] = ijk2idx(im1, jp2, k);
            val_ch_csr[idx2 + 36] = ak * gt;

            //---------------------------------------

            idx_ch_csr[idx2 + 37] = ijk2idx(ip2, jp1, k);
            val_ch_csr[idx2 + 37] = -ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 38] = ijk2idx(im2, jm1, k);
            val_ch_csr[idx2 + 38] = -ak * gt * (1. + gt2);

            //-----------------

            idx_ch_csr[idx2 + 39] = ijk2idx(ip2, jm1, k);
            val_ch_csr[idx2 + 39] = ak * gt * (1. + gt2);

            idx_ch_csr[idx2 + 40] = ijk2idx(im2, jp1, k);
            val_ch_csr[idx2 + 40] = ak * gt * (1. + gt2);

            //---------------------------------------

            idx_ch_csr[idx2 + 41] = ijk2idx(ip2, jp2, k);
            val_ch_csr[idx2 + 41] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 42] = ijk2idx(im2, jm2, k);
            val_ch_csr[idx2 + 42] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 43] = ijk2idx(ip2, jm2, k);
            val_ch_csr[idx2 + 43] = ak * gt2 * 0.25;

            idx_ch_csr[idx2 + 44] = ijk2idx(im2, jp2, k);
            val_ch_csr[idx2 + 44] = ak * gt2 * 0.25;
        }
    }

    // const vector
#pragma omp parallel for
    for (int idx = is_ch; idx < ie_ch; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);

        int ip1 = adj(1, i, NX);
        int jp1 = adj(1, j, NY);
        int kp1 = adj(1, k, NZ);

        int im1 = adj(-1, i, NX);
        int jm1 = adj(-1, j, NY);
        int km1 = adj(-1, k, NZ);

        double psi_im   = psi[ijk2im(i, j, k)];
        double psi_o_im = psi_o[ijk2im(i, j, k)];

        double bs =
            0.5 * (4. * psi_im - psi_o_im) * INV_DT +
            2. * kiDX2 *
                ((1. + gt2) * (potential_deriv(psi[ijk2im(ip1, j, k)]) + potential_deriv(psi[ijk2im(im1, j, k)])) +
                 potential_deriv(psi[ijk2im(i, jp1, k)]) + potential_deriv(psi[ijk2im(i, jm1, k)]) +
                 potential_deriv(psi[ijk2im(i, j, kp1)]) + potential_deriv(psi[ijk2im(i, j, km1)]) -
                 0.5 * gt *
                     (potential_deriv(psi[ijk2im(ip1, jp1, k)]) - potential_deriv(psi[ijk2im(ip1, jm1, k)]) -
                      potential_deriv(psi[ijk2im(im1, jp1, k)]) + potential_deriv(psi[ijk2im(im1, jm1, k)])) -
                 2. * (3. + gt2) * potential_deriv(psi_im)) -
            kiDX2 *
                ((1. + gt2) * (potential_deriv(psi_o[ijk2im(ip1, j, k)]) + potential_deriv(psi_o[ijk2im(im1, j, k)])) +
                 potential_deriv(psi_o[ijk2im(i, jp1, k)]) + potential_deriv(psi_o[ijk2im(i, jm1, k)]) +
                 potential_deriv(psi_o[ijk2im(i, j, kp1)]) + potential_deriv(psi_o[ijk2im(i, j, km1)]) -
                 0.5 * gt *
                     (potential_deriv(psi_o[ijk2im(ip1, jp1, k)]) - potential_deriv(psi_o[ijk2im(ip1, jm1, k)]) -
                      potential_deriv(psi_o[ijk2im(im1, jp1, k)]) + potential_deriv(psi_o[ijk2im(im1, jm1, k)])) -
                 2. * (3. + gt2) * potential_deriv(psi_o_im)) +
            ps.w * A_XI * kiDX2 *
                ((1. + gt2) * (calc_gradient_norm_OBL(phi, ijk2im(ip1, j, k), gt) +
                               calc_gradient_norm_OBL(phi, ijk2im(im1, j, k), gt)) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, jp1, k), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, jm1, k), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, j, kp1), gt) +
                 calc_gradient_norm_OBL(phi, ijk2im(i, j, km1), gt) -
                 0.5 * gt *
                     (calc_gradient_norm_OBL(phi, ijk2im(ip1, jp1, k), gt) -
                      calc_gradient_norm_OBL(phi, ijk2im(ip1, jm1, k), gt) -
                      calc_gradient_norm_OBL(phi, ijk2im(im1, jp1, k), gt) +
                      calc_gradient_norm_OBL(phi, ijk2im(im1, jm1, k), gt)) -
                 2. * (3. + gt2) * calc_gradient_norm_OBL(phi, ijk2im(i, j, k), gt)) -
            2. * ps.neutral * ps.d * kiDX2 *
                ((1. + gt2) * (phi[ijk2im(ip1, j, k)] + phi[ijk2im(im1, j, k)]) + phi[ijk2im(i, jp1, k)] +
                 phi[ijk2im(i, jm1, k)] + phi[ijk2im(i, j, kp1)] + phi[ijk2im(i, j, km1)] -
                 0.5 * gt *
                     (phi[ijk2im(ip1, jp1, k)] - phi[ijk2im(ip1, jm1, k)] - phi[ijk2im(im1, jp1, k)] +
                      phi[ijk2im(im1, jm1, k)]) -
                 2. * (3. + gt2) * phi[ijk2im(i, j, k)]);

#ifdef _LIS_SOLVER
        lis_vector_set_value(LIS_INS_VALUE, idx, bs, b_ch);
#else
        b_ch[idx] = bs;
#endif
    }
    // solver
#ifdef _LIS_SOLVER
    lis_matrix_set_csr(nval * 45, ptr_ch, idx_ch_csr, val_ch_csr, A_ch);
    lis_matrix_assemble(A_ch);
    lis_solve(A_ch, b_ch, x_ch, lis_solver_ch);
#else
    bicgstab(idx_ch_csr, val_ch_csr, ptr_ch, b_ch, wm_ch, nval);
#endif

    Cpy_v1(psi_o, psi);

    // set solutions
#pragma omp parallel for
    for (int idx = 0; idx < nval; idx++) {
        int i, j, k;
        idx2ijk(idx, &i, &j, &k);
        int im = ijk2im(i, j, k);
#ifdef _LIS_SOLVER
        psi[im] = x_ch->value[idx];
#else
        psi[im] = wm_ch.x[ijk2idx(i, j, k)];
#endif
    }
}
#ifdef _LIS_SOLVER
void Mem_alloc_lis(void) {
    int nnzval_ns, nnzval_ch;
    int nval_ns = NX * NY * NZ * DIM;
    int nval_ch = NX * NY * NZ;

    if (SW_EQ == Navier_Stokes_FDM) {
        nnzval_ns = nval_ns * 7;
    } else if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
        if (VISCOSITY_CHANGE) {
            nnzval_ns = nval_ns * 11;
        } else {
            nnzval_ns = nval_ns * 7;
        }
    } else if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
        if (VISCOSITY_CHANGE) {
            nnzval_ns = nval_ns * 20;
        } else {
            nnzval_ns = nval_ns * 12;
        }
    }

    if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
        nnzval_ch = nval_ch * 25;
    } else if (SW_EQ == Shear_NS_LE_CH_FDM) {
        nnzval_ch = nval_ch * 45;
    }

    lis_matrix_malloc_csr(NX * NY * NZ * DIM, nnzval_ns, &ptr_ns, &idx_ns_csr, &val_ns_csr);
    lis_matrix_create(LIS_COMM_WORLD, &A_ns);
    lis_matrix_set_size(A_ns, NX * NY * NZ * DIM, 0);

    lis_matrix_malloc_csr(NX * NY * NZ, nnzval_ch, &ptr_ch, &idx_ch_csr, &val_ch_csr);
    lis_matrix_create(LIS_COMM_WORLD, &A_ch);
    lis_matrix_set_size(A_ch, NX * NY * NZ, 0);
}
void Free_lis(void) {
    lis_matrix_destroy(A_ns);
    lis_vector_destroy(b_ns);
    lis_vector_destroy(x_ns);
    lis_solver_destroy(lis_solver_ns);

    lis_matrix_destroy(A_ch);
    lis_vector_destroy(b_ch);
    lis_vector_destroy(x_ch);
    lis_solver_destroy(lis_solver_ch);
}

void Init_lis(int argc, char *argv[]) {
    lis_initialize(&argc, &argv);

    lis_solver_create(&lis_solver_ns);
    lis_solver_set_option("-i 4 -p 0 -initx_zeros 1 -tol 1.0e-9 -maxiter 1000 -f 0 -print 0",
                          lis_solver_ns);  // -i 4 BiCGSTAB, 6 GPBiCG, 9 GMRES see Lis Documents
    lis_vector_duplicate(A_ns, &b_ns);
    lis_vector_duplicate(A_ns, &x_ns);
    lis_matrix_get_range(A_ns, &is_ns, &ie_ns);

    lis_solver_create(&lis_solver_ch);
    lis_solver_set_option("-i 4 -p 0 -initx_zeros 1 -tol 1.0e-9 -maxiter 1000 -f 0 -print 0", lis_solver_ch);
    lis_vector_duplicate(A_ch, &b_ch);
    lis_vector_duplicate(A_ch, &x_ch);
    lis_matrix_get_range(A_ch, &is_ch, &ie_ch);
}
#else
void Mem_alloc_matrix_solver(void) {
    if (SW_NSST == implicit_scheme) {
        int nval_ns = NX * NY * NZ * DIM;
        int nnzval;
        if (SW_EQ == Navier_Stokes_FDM) {
            nnzval = nval_ns * 7;
        } else if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
            if (VISCOSITY_CHANGE) {
                nnzval = nval_ns * 11;
            } else {
                nnzval = nval_ns * 7;
            }
        } else if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
            if (VISCOSITY_CHANGE) {
                nnzval = nval_ns * 20;
            } else {
                nnzval = nval_ns * 12;
            }
        }

        idx_ns_csr = calloc_1d_int(nnzval);
        val_ns_csr = calloc_1d_double(nnzval);
        ptr_ns     = calloc_1d_int(nval_ns + 1);
        b_ns       = calloc_1d_double(nval_ns);

        wm_ns.p   = calloc_1d_double(nval_ns);
        wm_ns.r   = calloc_1d_double(nval_ns);
        wm_ns.t   = calloc_1d_double(nval_ns);
        wm_ns.Ap  = calloc_1d_double(nval_ns);
        wm_ns.At  = calloc_1d_double(nval_ns);
        wm_ns.r0s = calloc_1d_double(nval_ns);
        wm_ns.x   = calloc_1d_double(nval_ns);
        wm_ns.Ax  = calloc_1d_double(nval_ns);
    }

    if (SW_CHST == implicit_scheme) {
        int nval_ch = NX * NY * NZ;
        int nnzval;
        if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
            nnzval = nval_ch * 25;
        } else if (SW_EQ == Shear_NS_LE_CH_FDM) {
            nnzval = nval_ch * 45;
        }

        idx_ch_csr = calloc_1d_int(nnzval);
        val_ch_csr = calloc_1d_double(nnzval);
        ptr_ch     = calloc_1d_int(nval_ch + 1);
        b_ch       = calloc_1d_double(nval_ch);

        wm_ch.p   = calloc_1d_double(nval_ch);
        wm_ch.r   = calloc_1d_double(nval_ch);
        wm_ch.t   = calloc_1d_double(nval_ch);
        wm_ch.Ap  = calloc_1d_double(nval_ch);
        wm_ch.At  = calloc_1d_double(nval_ch);
        wm_ch.r0s = calloc_1d_double(nval_ch);
        wm_ch.x   = calloc_1d_double(nval_ch);
        wm_ch.Ax  = calloc_1d_double(nval_ch);
    }
}

void Free_matrix_solver(void) {
    if (SW_NSST == implicit_scheme) {
        free_1d_int(idx_ns_csr);
        free_1d_double(val_ns_csr);
        free_1d_int(ptr_ns);
        free_1d_double(b_ns);

        free_1d_double(wm_ns.p);
        free_1d_double(wm_ns.r);
        free_1d_double(wm_ns.t);
        free_1d_double(wm_ns.Ap);
        free_1d_double(wm_ns.At);
        free_1d_double(wm_ns.r0s);
        free_1d_double(wm_ns.x);
        free_1d_double(wm_ns.Ax);
    }
    if (SW_CHST == implicit_scheme) {
        free_1d_int(idx_ch_csr);
        free_1d_double(val_ch_csr);
        free_1d_int(ptr_ch);
        free_1d_double(b_ch);

        free_1d_double(wm_ch.p);
        free_1d_double(wm_ch.r);
        free_1d_double(wm_ch.t);
        free_1d_double(wm_ch.Ap);
        free_1d_double(wm_ch.At);
        free_1d_double(wm_ch.r0s);
        free_1d_double(wm_ch.x);
        free_1d_double(wm_ch.Ax);
    }
}

void bicgstab(int *idx, double *val, int *row_ptr, double *b, work_bicgstab &wm, int nval) {
    double alpha, beta, zeta, r0sr, r0sr_o, r0sAp, tAt, AtAt;
    double err, rr, bb;

    int ITER = 0;
#pragma omp parallel for
    for (int i = 0; i < nval; i++) {
        wm.p[i]   = 0.;
        wm.r[i]   = 0.;
        wm.t[i]   = 0.;
        wm.Ap[i]  = 0.;
        wm.At[i]  = 0.;
        wm.r0s[i] = 0.;
        wm.x[i]   = 0.;
        wm.Ax[i]  = 0.;
    }
    bb    = 0.;
    alpha = 0.;
    beta  = 0.;
    r0sr  = 0.;
    zeta  = 0.;

#pragma omp parallel for reduction(+ : r0sr, bb)
    for (int i = 0; i < nval; i++) {
        wm.r[i] = b[i] - wm.Ax[i];
        bb += b[i] * b[i];
        wm.r0s[i] = wm.r[i];
        r0sr += wm.r[i] * wm.r0s[i];
    }
    bb = sqrt(bb);
    if (bb > 0.) {
        do {
#pragma omp parallel for
            for (int i = 0; i < nval; i++) {
                wm.p[i] = wm.r[i] + beta * (wm.p[i] - zeta * wm.Ap[i]);
            }
            Calc_Ax(wm.p, idx, val, row_ptr, wm.Ap, nval);
            r0sAp = 0.;
#pragma omp parallel for reduction(+ : r0sAp)
            for (int i = 0; i < nval; i++) {
                r0sAp += wm.r0s[i] * wm.Ap[i];
            }
            alpha = r0sr / r0sAp;

#pragma omp parallel for
            for (int i = 0; i < nval; i++) {
                wm.t[i] = wm.r[i] - alpha * wm.Ap[i];
            }

            Calc_Ax(wm.t, idx, val, row_ptr, wm.At, nval);
            tAt  = 0.;
            AtAt = 0.;
#pragma omp parallel for reduction(+ : tAt, AtAt)
            for (int i = 0; i < nval; i++) {
                tAt += wm.t[i] * wm.At[i];
                AtAt += wm.At[i] * wm.At[i];
            }
            zeta = tAt / AtAt;

            r0sr_o = r0sr;
            r0sr   = 0.;
            rr     = 0.;
#pragma omp parallel for reduction(+ : r0sr, rr)
            for (int i = 0; i < nval; i++) {
                wm.x[i] += alpha * wm.p[i] + zeta * wm.t[i];
                wm.r[i] = wm.t[i] - zeta * wm.At[i];
                r0sr += wm.r0s[i] * wm.r[i];
                rr += wm.r[i] * wm.r[i];
            }
            err  = log10(rr) / 2. - log10(bb);
            beta = r0sr / (zeta * r0sAp);
            ITER++;
        } while (err > wm.eps && ITER < wm.maxiter);
    }

    if (ITER == wm.maxiter) {
        fprintf(stderr, "Error: BiCGSTAB method is not converged \n");
        exit_job(EXIT_FAILURE);
    } else {
        // fprintf(stderr, "ITER: %d \n", ITER);
    }
}
void Init_ns(void) {
    wm_ns.eps     = eps_ns;
    wm_ns.maxiter = maxiter_ns;
    fprintf(stderr, "# NS iter setting: %f %d\n", wm_ns.eps, wm_ns.maxiter);
}
void Init_ch(void) {
    wm_ch.eps     = eps_ch;
    wm_ch.maxiter = maxiter_ch;
    fprintf(stderr, "# CH iter setting: %f %d\n", wm_ch.eps, wm_ch.maxiter);
}
#endif
