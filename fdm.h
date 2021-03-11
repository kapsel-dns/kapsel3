#ifndef FDM_H
#define FDM_H

#include "fdm_matrix_solver.h"
#include "fdm_phase_separation.h"
#include "fft_wrapper.h"
#include "fluid_solver.h"
#include "input.h"
#include "make_phi.h"

extern double **adv;
extern double **adv_o;
extern double **lap;
extern double **lap_o;
extern double * rhs;
extern double **u_s;
extern double **u_o;
extern double **u_o_cpy;
extern double * w_v1;
extern double **w_v3;
extern double **w_v3_2;
extern double **w_v3_3;
extern double * eta_s;
extern double * shear_rate_field;

extern double sreff_old;

inline void dc_offset(double *u) {
    int    im;
    double sum_u = 0.;
#pragma omp parallel for private(im) reduction(+ : sum_u)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                sum_u += u[im];
            }
        }
    }
    sum_u /= (double)(NX * NY * NZ);
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                u[im] -= sum_u;
            }
        }
    }
}

inline int adj(int x, int i, int NX) { return ((i + x) < 0) ? i + x + NX : (((i + x) > NX - 1) ? i + x - NX : i + x); }

inline int ijk2im(int i, int j, int k) { return (i * NY * NZ_) + (j * NZ_) + k; }

inline void im2ijk(int im, int *i, int *j, int *k) {
    int im_dmy = im;
    *i         = im_dmy / (NY * NZ_);
    im_dmy -= *i * NY * NZ_;
    *j = im_dmy / NZ_;
    im_dmy -= *j * NZ_;
    *k = im_dmy;
}

inline int ijkd2idx(int i, int j, int k, int d) {
    // i, j, k, d -> idx
    return d * (NX * NY * NZ) + i * (NY * NZ) + j * NZ + k;
}

inline void idx2ijkd(int idx, int *i, int *j, int *k, int *d) {
    // idx -> i, j, k, d
    int dmy = idx;
    *d      = dmy / (NX * NY * NZ);
    dmy -= (NX * NY * NZ) * *d;
    *i = dmy / (NY * NZ);
    dmy -= (NY * NZ) * *i;
    *j = dmy / NZ;
    *k = dmy - NZ * *j;
}

inline void idx2ijk(int idx, int *i, int *j, int *k) {
    // idx -> i, j, k
    int dmy = idx;
    *i      = dmy / (NY * NZ);
    dmy -= (NY * NZ) * *i;
    *j = dmy / NZ;
    *k = dmy - NZ * *j;
}

inline int ijk2idx(int i, int j, int k) {
    // i, j, k -> idx
    return i * (NY * NZ) + j * NZ + k;
}

inline double calc_gradient_o1_to_o1(const double *field, int im, int dir) {
    const double INV_2DX = 1. / (2. * DX);
    int          i, j, k;

    im2ijk(im, &i, &j, &k);

    // periodic boundary condition
    int ip1, jp1, kp1;
    int im1, jm1, km1;

    ip1 = adj(1, i, NX);
    jp1 = adj(1, j, NY);
    kp1 = adj(1, k, NZ);

    im1 = adj(-1, i, NX);
    jm1 = adj(-1, j, NY);
    km1 = adj(-1, k, NZ);

    // set adjacent meshes
    int im_ip1 = ijk2im(ip1, j, k);
    int im_jp1 = ijk2im(i, jp1, k);
    int im_kp1 = ijk2im(i, j, kp1);

    int im_im1 = ijk2im(im1, j, k);
    int im_jm1 = ijk2im(i, jm1, k);
    int im_km1 = ijk2im(i, j, km1);

    double retval = 0;
    switch (dir) {
        case 0:
            retval = (field[im_ip1] - field[im_im1]) * INV_2DX;
            break;
        case 1:
            retval = (field[im_jp1] - field[im_jm1]) * INV_2DX;
            break;
        case 2:
            retval = (field[im_kp1] - field[im_km1]) * INV_2DX;
            break;
        default:
            std::cout << "calc_gradient_o1_to_o1_error" << std::endl;
            break;
    }
    return retval;
}

inline double calc_gradient_o1_to_o2(const double *field, int im, int dir) {
    const double INV_4DX = 1. / (4. * DX);
    int          i, j, k;
    im2ijk(im, &i, &j, &k);

    // periodic boundary condition
    int ip1, jp1, kp1;

    ip1 = adj(1, i, NX);
    jp1 = adj(1, j, NY);
    kp1 = adj(1, k, NZ);

    double retval = 0;
    switch (dir) {
        case 0:
            retval = (field[ijk2im(ip1, jp1, kp1)] - field[ijk2im(i, jp1, kp1)] + field[ijk2im(ip1, jp1, k)] -
                      field[ijk2im(i, jp1, k)] + field[ijk2im(ip1, j, kp1)] - field[ijk2im(i, j, kp1)] +
                      field[ijk2im(ip1, j, k)] - field[im]) *
                     INV_4DX;
            break;
        case 1:
            retval = (field[ijk2im(ip1, jp1, kp1)] - field[ijk2im(ip1, j, kp1)] + field[ijk2im(ip1, jp1, k)] -
                      field[ijk2im(ip1, j, k)] + field[ijk2im(i, jp1, kp1)] - field[ijk2im(i, j, kp1)] +
                      field[ijk2im(i, jp1, k)] - field[im]) *
                     INV_4DX;
            break;
        case 2:
            retval = (field[ijk2im(ip1, jp1, kp1)] - field[ijk2im(ip1, jp1, k)] + field[ijk2im(ip1, j, kp1)] -
                      field[ijk2im(ip1, j, k)] + field[ijk2im(i, jp1, kp1)] - field[ijk2im(i, jp1, k)] +
                      field[ijk2im(i, j, kp1)] - field[im]) *
                     INV_4DX;
            break;
        default:
            std::cout << "calc_gradient_o1_to_o2_error" << std::endl;
            break;
    }
    return retval;
}

inline double calc_gradient_o2_to_o1(const double *field, int im, int dir) {
    const double INV_4DX = 1. / (4. * DX);
    int          i, j, k;
    im2ijk(im, &i, &j, &k);

    // periodic boundary condition
    int im1, jm1, km1;

    im1 = adj(-1, i, NX);
    jm1 = adj(-1, j, NY);
    km1 = adj(-1, k, NZ);

    double retval = 0;
    switch (dir) {
        case 0:
            retval = (field[im] - field[ijk2im(im1, j, k)] + field[ijk2im(i, jm1, k)] - field[ijk2im(im1, jm1, k)] +
                      field[ijk2im(i, j, km1)] - field[ijk2im(im1, j, km1)] + field[ijk2im(i, jm1, km1)] -
                      field[ijk2im(im1, jm1, km1)]) *
                     INV_4DX;
            break;
        case 1:
            retval = (field[im] - field[ijk2im(i, jm1, k)] + field[ijk2im(im1, j, k)] - field[ijk2im(im1, jm1, k)] +
                      field[ijk2im(i, j, km1)] - field[ijk2im(i, jm1, km1)] + field[ijk2im(im1, j, km1)] -
                      field[ijk2im(im1, jm1, km1)]) *
                     INV_4DX;
            break;
        case 2:
            retval = (field[im] - field[ijk2im(i, j, km1)] + field[ijk2im(im1, j, k)] - field[ijk2im(im1, j, km1)] +
                      field[ijk2im(i, jm1, k)] - field[ijk2im(i, jm1, km1)] + field[ijk2im(im1, jm1, k)] -
                      field[ijk2im(im1, jm1, km1)]) *
                     INV_4DX;
            break;
        default:
            std::cout << "calc_gradient_o2_to_o1_error" << std::endl;
            break;
    }
    return retval;
}

inline double calc_laplacian(const double *field, int im) {
    const double INV_DX2 = 1. / (DX * DX);
    int          i, j, k;

    im2ijk(im, &i, &j, &k);
    // periodic boundary condition
    int ip1, jp1, kp1;
    int im1, jm1, km1;

    ip1 = adj(1, i, NX);
    jp1 = adj(1, j, NY);
    kp1 = adj(1, k, NZ);

    im1 = adj(-1, i, NX);
    jm1 = adj(-1, j, NY);
    km1 = adj(-1, k, NZ);

    // set adjacent meshes
    int im_ip1 = ijk2im(ip1, j, k);
    int im_jp1 = ijk2im(i, jp1, k);
    int im_kp1 = ijk2im(i, j, kp1);

    int im_im1 = ijk2im(im1, j, k);
    int im_jm1 = ijk2im(i, jm1, k);
    int im_km1 = ijk2im(i, j, km1);

    double x_dir = (field[im_ip1] - 2. * field[im] + field[im_im1]);
    double y_dir = (field[im_jp1] - 2. * field[im] + field[im_jm1]);
    double z_dir = (field[im_kp1] - 2. * field[im] + field[im_km1]);

    return (x_dir + y_dir + z_dir) * INV_DX2;
}

inline double calc_laplacian_OBL(const double *field, int im, const double degree_oblique) {
    const double INV_DX2  = 1. / (DX * DX);
    const double INV_4DX2 = 1. / (4. * DX * DX);
    int          i, j, k;

    im2ijk(im, &i, &j, &k);
    // periodic boundary condition
    int ip1 = adj(1, i, NX);
    int jp1 = adj(1, j, NY);
    int kp1 = adj(1, k, NZ);

    int im1 = adj(-1, i, NX);
    int jm1 = adj(-1, j, NY);
    int km1 = adj(-1, k, NZ);

    int im_ip1_jp1 = ijk2im(ip1, jp1, k);
    int im_ip1_jm1 = ijk2im(ip1, jm1, k);
    int im_im1_jp1 = ijk2im(im1, jp1, k);
    int im_im1_jm1 = ijk2im(im1, jm1, k);

    // set adjacent meshes
    int im_ip1 = ijk2im(ip1, j, k);
    int im_jp1 = ijk2im(i, jp1, k);
    int im_kp1 = ijk2im(i, j, kp1);

    int im_im1 = ijk2im(im1, j, k);
    int im_jm1 = ijk2im(i, jm1, k);
    int im_km1 = ijk2im(i, j, km1);

    double x_dir = (field[im_ip1] - 2. * field[im] + field[im_im1]) * INV_DX2;
    double y_dir = (field[im_jp1] - 2. * field[im] + field[im_jm1]) * INV_DX2;
    double z_dir = (field[im_kp1] - 2. * field[im] + field[im_km1]) * INV_DX2;

    double mixed_deriv = (field[im_ip1_jp1] - field[im_ip1_jm1] - field[im_im1_jp1] + field[im_im1_jm1]) * INV_4DX2;

    return (1. + degree_oblique * degree_oblique) * x_dir - (2. * degree_oblique * mixed_deriv) + y_dir + z_dir;
}

inline double calc_gradient_norm(double *phi, int im) {
    double dphi_dx = calc_gradient_o1_to_o1(phi, im, 0);
    double dphi_dy = calc_gradient_o1_to_o1(phi, im, 1);
    double dphi_dz = calc_gradient_o1_to_o1(phi, im, 2);

    double grad_phi_norm = dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz;

    return grad_phi_norm;
}

inline double calc_gradient_norm_sqrt(double *phi, int im) { return sqrt(calc_gradient_norm(phi, im)); }

inline double calc_gradient_norm_OBL(double *phi, int im, const double gt) {
    double dphi_dx_co = calc_gradient_o1_to_o1(phi, im, 0);
    double dphi_dy_co = calc_gradient_o1_to_o1(phi, im, 1);
    double dphi_dz_co = calc_gradient_o1_to_o1(phi, im, 2);

    double dphi_dx_contra = ((1. + gt * gt) * dphi_dx_co) - (gt * dphi_dy_co);
    double dphi_dy_contra = -(gt * dphi_dx_co) + dphi_dy_co;
    double dphi_dz_contra = dphi_dz_co;

    double grad_phi_norm = dphi_dx_co * dphi_dx_contra + dphi_dy_co * dphi_dy_contra + dphi_dz_co * dphi_dz_contra;

    return grad_phi_norm;
}

inline void Solve_poisson_dft(double *p, double *s) {
    A2a_k(s);
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                p[im] = -IK2[im] * s[im];
            }
        }
    }
    A_k2a(p);
}

inline void Cpy_v3(double **dst, double **src) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im         = (i * NY * NZ_) + (j * NZ_) + k;
                dst[0][im] = src[0][im];
                dst[1][im] = src[1][im];
                dst[2][im] = src[2][im];
            }
        }
    }
}
inline void Cpy_v1(double *dst, double *src) {
    int im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im      = (i * NY * NZ_) + (j * NZ_) + k;
                dst[im] = src[im];
            }
        }
    }
}
inline void Set_poisson_rhs_sub(double **u, double **dmy, double *s, CTime &jikan) {
    const double INV_DT = 1. / jikan.dt_fluid;
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                double dux_dx = calc_gradient_o1_to_o2(u[0], im, 0);
                double duy_dy = calc_gradient_o1_to_o2(u[1], im, 1);
                double duz_dz = calc_gradient_o1_to_o2(u[2], im, 2);

                double drx_dx = calc_gradient_o1_to_o2(dmy[0], im, 0);
                double dry_dy = calc_gradient_o1_to_o2(dmy[1], im, 1);
                double drz_dz = calc_gradient_o1_to_o2(dmy[2], im, 2);

                s[im] = (RHO * INV_DT) * (dux_dx + duy_dy + duz_dz) - RHO * (drx_dx + dry_dy + drz_dz);
            }
        }
    }
}

inline void Calc_ab2_val(double *retval, const double *new_val, const double *old_val) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im     = (i * NY * NZ_) + (j * NZ_) + k;
                retval[im] = 0.5 * (3. * new_val[im] - old_val[im]);
            }
        }
    }
}

inline void Make_phi_s(double *      phi,
                       double *      phi_sum,
                       Particle *    p,
                       const double &dx,
                       const int &   np_domain,
                       int **        sekibun_cell,
                       const int     Nlattice[DIM],
                       CTime &       jikan,
                       const double  radius = RADIUS) {
#pragma omp parallel for
    for (int n = 0; n < Particle_Number; n++) {
        double xp[DIM];
        for (int d = 0; d < DIM; d++) {
            if (jikan.ts == 0) {
                xp[d] = p[n].x[d];
            } else {
                xp[d] = (3. * p[n].x[d] - p[n].x_previous[d]) / 2.;
            }
        }

        PBC(xp);

        int    x_int[DIM];
        double residue[DIM];
        int    sw_in_cell = Particle_cell(xp, dx, x_int, residue);
        sw_in_cell        = 1;

        int    r_mesh[DIM];
        double dmy, dmy_phi;
        double r[DIM], x[DIM];
        for (int mesh = 0; mesh < np_domain; mesh++) {
            Relative_coord(sekibun_cell[mesh], x_int, residue, sw_in_cell, Nlattice, dx, r_mesh, r);

            for (int d = 0; d < DIM; d++) x[d] = r_mesh[d] * DX;

            dmy     = Distance(x, xp);
            dmy_phi = Phi(dmy, radius);

#pragma omp atomic
            phi_sum[(r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2]] += dmy_phi;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        int im;
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im      = (i * NY * NZ_) + (j * NZ_) + k;
                phi[im] = MIN(phi_sum[im], 1.0);
            }
        }
    }
}

inline int PBC_OBL_half(double *x, double &delta_vx, const double gt, const double sreff) {
    double signY = x[1];
    x[1]         = fmod(x[1] + L_particle[1], L_particle[1]);
    signY -= x[1];
    int sign = (int)signY;
    if (!(sign == 0)) {
        sign = sign / abs(sign);
    }

    x[0] -= (double)sign * gt * L_particle[1];
    x[0] = fmod(x[0] + L_particle[0], L_particle[0]);
    x[2] = fmod(x[2] + L_particle[2], L_particle[2]);
    for (int d = 0; d < DIM; d++) {
        assert(x[d] >= 0);
        assert(x[d] < L[d]);
    }

    delta_vx = -(double)sign * sreff * L_particle[1];
    return sign;
}

inline void Make_phi_s_OBL(double *      phi,
                           double *      phi_sum,
                           Particle *    p,
                           const double &dx,
                           const int &   np_domain,
                           int **        sekibun_cell,
                           const int     Nlattice[DIM],
                           const double  gt,
                           const double  sreff,
                           CTime &       jikan,
                           const double  radius = RADIUS) {
#pragma omp parallel for
    for (int n = 0; n < Particle_Number; n++) {
        double xp[DIM];
        double dmyy;
        for (int d = 0; d < DIM; d++) {
            if (jikan.ts == 0) {
                xp[d] = p[n].x[d];
            } else {
                xp[d] = (3. * p[n].x[d] - p[n].x_previous[d]) / 2.;
            }
        }
        int dmysign = PBC_OBL_half(xp, dmyy, gt, sreff);

        int    x_int[DIM];
        double residue[DIM];
        int    sw_in_cell = Particle_cell(xp, dx, x_int, residue);
        sw_in_cell        = 1;

        int    sign;
        int    r_mesh[DIM];
        double dmy, dmy_phi;
        double r[DIM], x[DIM];
        for (int mesh = 0; mesh < np_domain; mesh++) {
            sign = Relative_coord_check_stepover_Y(
                sekibun_cell[mesh], x_int, residue, sw_in_cell, Nlattice, dx, r_mesh, r);

            for (int d = 0; d < DIM; d++) x[d] = r_mesh[d] * dx;

            dmy     = Distance_OBL(x, xp);
            dmy_phi = Phi(dmy, radius);

#pragma omp atomic
            phi_sum[(r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2]] += dmy_phi;
        }
    }
    {
#pragma omp parallel for
        for (int i = 0; i < NX; i++) {
            int im;

            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < NZ; k++) {
                    im      = (i * NY * NZ_) + (j * NZ_) + k;
                    phi[im] = MIN(phi_sum[im], 1.0);
                }
            }
        }
    }
}

inline void A2a_oblique(double *a) {
    int im;
    int im_ob;
    int im_ob_p;

    Copy_v1(w_v1, a);
#pragma omp parallel for private(im, im_ob, im_ob_p)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int    i_oblique, i_oblique_plus;
            double alpha, beta;
            orth2obl(j, i, i_oblique, i_oblique_plus, alpha, beta);

            for (int k = 0; k < NZ; k++) {
                im      = (i * NY * NZ_) + (j * NZ_) + k;
                im_ob   = (i_oblique * NY * NZ_) + (j * NZ_) + k;
                im_ob_p = (i_oblique_plus * NY * NZ_) + (j * NZ_) + k;
                a[im]   = beta * w_v1[im_ob] + alpha * w_v1[im_ob_p];
            }
        }
    }
}

inline void A_oblique2a(double *a) {
    int im;
    int im_ob;
    int im_p;

    Copy_v1(work_v1, a);

#pragma omp parallel for private(im, im_ob, im_p)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int    i_plus, i_oblique;
            double alpha, beta;
            obl2orth(j, i, i_plus, i_oblique, alpha, beta);

            for (int k = 0; k < NZ; k++) {
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                im_ob = (i_oblique * NY * NZ_) + (j * NZ_) + k;
                im_p  = (i_plus * NY * NZ_) + (j * NZ_) + k;

                a[im_ob] = beta * work_v1[im] + alpha * work_v1[im_p];
            }
        }
    }
}

inline void A_oblique2a_out(double *a, double *out) {
    int im;
    int im_ob;
    int im_p;

#pragma omp parallel for private(im, im_ob, im_p)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int    i_plus, i_oblique;
            double alpha, beta;
            obl2orth(j, i, i_plus, i_oblique, alpha, beta);

            for (int k = 0; k < NZ; k++) {
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                im_ob = (i_oblique * NY * NZ_) + (j * NZ_) + k;
                im_p  = (i_plus * NY * NZ_) + (j * NZ_) + k;

                out[im_ob] = beta * a[im] + alpha * a[im_p];
            }
        }
    }
}

inline void Reset_A_OBL(double *a, double const *acp, const int &flg) {
    const double delta = (flg == 0 ? XYaspect : -XYaspect);
    int          im, im_obl;

#pragma omp parallel for private(im, im_obl)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double sign = j - NY / 2;
            if (!(sign == 0)) {
                sign = sign / fabs(sign);
            }

            int i_oblique = (int)(sign * (j - NY / 2)) * sign;
            i_oblique     = (int)fmod(i + delta * i_oblique + 4. * NX, NX);
            for (int k = 0; k < NZ; k++) {
                im        = (i * NY * NZ_) + (j * NZ_) + k;
                im_obl    = (i_oblique * NY * NZ_) + (j * NZ_) + k;
                a[im_obl] = acp[im];
            }
        }
    }
}
inline void Spline_A_oblique_transform(double *a, const OBL_TRANSFORM &flag) {
    int    im;
    double dmy_x;
    double delta_y;
    double sign;
    if (flag == oblique2cartesian) {
        sign = -1.0;
    } else if (flag == cartesian2oblique) {
        sign = 1.0;
    } else {
        exit_job(EXIT_FAILURE);
    }

#pragma omp parallel for private(im, dmy_x, delta_y)
    for (int j = 0; j < NY; j++) {  // original coord
        int np;
#ifndef _OPENMP
        np = 0;
#else
        np = omp_get_thread_num();
#endif
        splineSystem *spl = splineOblique[np];
        double **     us0 = uspline[np];

        delta_y = (double)(j - NY / 2) * DX;

        for (int k = 0; k < NZ; k++) {  // original coord

            // setup interpolation grid
            for (int i = 0; i < NX; i++) {  // original coord
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                dmy_x = fmod(i * DX - sign * degree_oblique * delta_y + 4.0 * LX, LX);  // transformed coord

                // velocity components in transformed basis defined over
                // original grid points x0
                us0[0][i] = a[im];
            }  // i

            // compute interpolated points
            splineCompute(spl, us0[0]);
            for (int i = 0; i < NX; i++) {  // transformed coord
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                dmy_x = fmod(i * DX + sign * degree_oblique * delta_y + 4.0 * LX, LX);  // original coord

                a[im] = splineFx(spl, dmy_x);
            }  // i
        }      // k
    }          // j
}

inline void Transform_obl_a(double *a, const OBL_TRANSFORM &flag) {
    if (SW_OBL_INT == linear_int) {
        if (flag == oblique2cartesian) {
            A_oblique2a(a);
        } else if (flag == cartesian2oblique) {
            A2a_oblique(a);
        } else {
            exit_job(EXIT_FAILURE);
        }
    } else if (SW_OBL_INT == spline_int) {
        Spline_A_oblique_transform(a, flag);
    } else {
        exit_job(EXIT_FAILURE);
    }
}

inline void Update_K2_OBL_hdt(const CTime &jikan) {
    const double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;
    int          im;
#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ_; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;

                K2[im] = SQ(WAVE_X * KX_int[im]) + SQ(WAVE_Y * KY_int[im] - WAVE_X * KX_int[im] * gt) +
                         SQ(WAVE_Z * KZ_int[im]);

                if (K2[im] > 0.0) {
                    IK2[im] = 1.0 / K2[im];
                } else {
                    IK2[im] = 0.0;
                }
            }
        }
    }
}

inline double calc_gamma(double **u, int im) {
    double dux_dx = calc_gradient_o1_to_o1(u[0], im, 0);
    double dux_dy = calc_gradient_o1_to_o1(u[0], im, 1) + Shear_rate;
    double dux_dz = calc_gradient_o1_to_o1(u[0], im, 2);

    double duy_dx = calc_gradient_o1_to_o1(u[1], im, 0);
    double duy_dy = calc_gradient_o1_to_o1(u[1], im, 1);
    double duy_dz = calc_gradient_o1_to_o1(u[1], im, 2);

    double duz_dx = calc_gradient_o1_to_o1(u[2], im, 0);
    double duz_dy = calc_gradient_o1_to_o1(u[2], im, 1);
    double duz_dz = calc_gradient_o1_to_o1(u[2], im, 2);

    double invariant_half = 2. * (dux_dx * dux_dx + duy_dy * duy_dy + duz_dz * duz_dz) +
                            (duy_dx + dux_dy) * (duy_dx + dux_dy) + (duz_dy + duy_dz) * (duz_dy + duy_dz) +
                            (dux_dz + duz_dx) * (dux_dz + duz_dx);

    return sqrt(invariant_half);
}

inline double calc_div_tensor_gradient_inner(const double *const *const *tensor, const double *field, int im) {
    const double INV_DX  = 1. / DX;
    const double INV_DX2 = 1. / (DX * DX);
    int          i, j, k;

    im2ijk(im, &i, &j, &k);

    // periodic boundary condition
    int ip1, jp1, kp1;
    int im1, jm1, km1;

    ip1 = adj(1, i, NX);
    jp1 = adj(1, j, NY);
    kp1 = adj(1, k, NZ);

    im1 = adj(-1, i, NX);
    jm1 = adj(-1, j, NY);
    km1 = adj(-1, k, NZ);

    double dxdx = (field[ijk2im(ip1, j, k)] + field[ijk2im(im1, j, k)] - 2. * field[im]) * INV_DX2;
    double dxdy =
        (field[ijk2im(i, jp1, k)] - field[im] - field[ijk2im(im1, jp1, k)] + field[ijk2im(im1, j, k)]) * INV_DX2;
    double dxdz =
        (field[ijk2im(i, j, kp1)] - field[im] - field[ijk2im(im1, j, kp1)] + field[ijk2im(im1, j, k)]) * INV_DX2;

    double dydx =
        (field[ijk2im(ip1, j, k)] - field[im] - field[ijk2im(ip1, jm1, k)] + field[ijk2im(i, jm1, k)]) * INV_DX2;
    double dydy = (field[ijk2im(i, jp1, k)] + field[ijk2im(i, jm1, k)] - 2. * field[im]) * INV_DX2;
    double dydz =
        (field[ijk2im(i, j, kp1)] - field[im] - field[ijk2im(i, jm1, kp1)] + field[ijk2im(i, jm1, k)]) * INV_DX2;

    double dzdx =
        (field[ijk2im(ip1, j, k)] - field[im] - field[ijk2im(ip1, j, km1)] + field[ijk2im(i, j, km1)]) * INV_DX2;
    double dzdy =
        (field[ijk2im(i, jp1, k)] - field[im] - field[ijk2im(i, jp1, km1)] + field[ijk2im(i, j, km1)]) * INV_DX2;
    double dzdz = (field[ijk2im(i, j, kp1)] + field[ijk2im(i, j, km1)] - 2. * field[im]) * INV_DX2;

    double x_term = tensor[0][0][im] * dxdx + tensor[0][1][im] * dxdy + tensor[0][2][im] * dxdz +
                    calc_gradient_o1_to_o1(tensor[0][0], im, 0) * calc_gradient_o1_to_o1(field, im, 0) +
                    calc_gradient_o1_to_o1(tensor[0][1], im, 0) * calc_gradient_o1_to_o1(field, im, 1) +
                    calc_gradient_o1_to_o1(tensor[0][2], im, 0) * calc_gradient_o1_to_o1(field, im, 2);
    double y_term = tensor[1][0][im] * dydx + tensor[1][1][im] * dydy + tensor[1][2][im] * dydz +
                    calc_gradient_o1_to_o1(tensor[1][0], im, 1) * calc_gradient_o1_to_o1(field, im, 0) +
                    calc_gradient_o1_to_o1(tensor[1][1], im, 1) * calc_gradient_o1_to_o1(field, im, 1) +
                    calc_gradient_o1_to_o1(tensor[1][2], im, 1) * calc_gradient_o1_to_o1(field, im, 2);
    double z_term = tensor[2][0][im] * dzdx + tensor[2][1][im] * dzdy + tensor[2][2][im] * dzdz +
                    calc_gradient_o1_to_o1(tensor[2][0], im, 2) * calc_gradient_o1_to_o1(field, im, 0) +
                    calc_gradient_o1_to_o1(tensor[2][1], im, 2) * calc_gradient_o1_to_o1(field, im, 1) +
                    calc_gradient_o1_to_o1(tensor[2][2], im, 2) * calc_gradient_o1_to_o1(field, im, 2);

    double retval = x_term + y_term + z_term;

    return retval;
}

void Make_psi_all(double *psi_all, double const *psi);

void NS_solver_slavedEuler_explicit(double **u, double *Pressure, Particle *p, CTime &jikan);
void NS_solver_slavedEuler_implicit(double **u, double *Pressure, Particle *p, CTime &jikan);
void NS_solver_slavedEuler_Shear_OBL_explicit(double **u, double *Pressure, Particle *p, CTime &jikan);
void NS_solver_slavedEuler_Shear_OBL_implicit(double **&u, double *Pressure, Particle *p, CTime &jikan);

void Set_poisson_rhs(double **u, double **adv_u, double **lap_u, double *s, CTime &jikan);

void Update_u_pressure(double **u, double *dp, CTime &jikan);
void Update_u_pressure_OBL(double **u, double *dp, CTime &jikan, const double degree_oblique);

void Update_u_adv_euler(double **u, double **adv_u, CTime &jikan);
void Update_u_adv_ab2(double **u, double **adv_u, double **adv_u_old, CTime &jikan);
void Update_u_lap_euler(double **u, double **lap_u, CTime &jikan);
void Update_u_lap_ab2(double **u, double **lap_u, double **lap_u_old, CTime &jikan);
void Update_p(double *p, double *dp);

void U2advection(double **u, double **adv_u);
void U2laplacian(double **u, double **lap_u);
void U2advection_OBL(double **u, double **adv_u, const double degree_oblique);
void U2laplacian_OBL(double **u, double **lap_u, const double degree_oblique);

void Mem_alloc_fdm(void);
void Free_fdm(void);

void calc_shear_rate_field(double **u, double *shear_rate_field);
#endif