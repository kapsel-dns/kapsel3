/*!
  \file fft_wrapper.h
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief FFT wrapper routines for reciprocal space calculations (header file)
  \note For simplicity, the equations given here for the various operations in Fourier space refer to the continuous
  transformation.
 */
#ifndef FFT_WRAPPER_H
#define FFT_WRAPPER_H

#include <assert.h>
#include <math.h>

#include <complex>

#include "alloc.h"
#include "aux_field.h"
#include "input.h"
#include "macro.h"
#include "periodic_spline.h"
#include "variable.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _FFT_IMKL
#include <mkl_dfti.h>
#elif _FFT_FFTW
#include <fftw3.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern void rdft3d(int n1, int n2, int n3, int sign, double ***a, double *t, int *ip, double *w);
extern void rdft3dsort(int, int, int, int, double ***);
#ifdef __cplusplus
}
#endif

////////////////////////
typedef std::complex<double> Complex;
/////////////  FFT
#ifdef _FFT_IMKL
extern DFTI_DESCRIPTOR_HANDLE imkl_p_fw, imkl_p_bw;
#elif _FFT_FFTW
extern fftw_plan fftw_p_fw, fftw_p_bw;
#endif
struct ooura_plan {
    int *     ip;
    double *  w;
    double *  t;
    double ***a;
};
extern ooura_plan ooura_p;

extern double **ucp;
extern double * phi, *phi_p, *phi_sum, **up, **u, *rhop, *phi_wall;
extern double **work_v3, **work_v2, *work_v1;

extern int *   KX_int, *KY_int, *KZ_int;
extern double *K2, *IK2;

extern splineSystem **splineOblique;
extern double ***     uspline;

extern Index_range *ijk_range_two_third_filter;
extern int          n_ijk_range_two_third_filter;

extern int (*Calc_KX)(const int &i, const int &j, const int &k);
extern int (*Calc_KY)(const int &i, const int &j, const int &k);
extern int (*Calc_KZ)(const int &i, const int &j, const int &k);
extern void (*Truncate_two_third_rule)(double *a);

/*!
  \brief Deallocate FFT memory
 */
void Free_fft(void);
/*!
  \brief Initialize FFT routines
  \details Used to initialize any required workspace memory for FFT routines. For the moment only Ooura's FFT can be
  used
 */
void Init_fft(void);

/*!
  \brief Compute x-derivative of scalar field (in reciprocal space)
  \details \f[
  \ft{A}(\vec{k}) \longrightarrow -i 2 \pi k_x \ft{A}(\vec{k}) = \fft{\partial_x A(\vec{r})}
  \f]
  \param[in] a Fourier transform of a scalar field A
  \param[out] da Fourier transform of x-derivative of A
 */
void A_k2dxa_k(double *a, double *da);

/*!
  \brief Compute y-derivative of scalar field (in reciprocal space)
  \details \f[
  \ft{A}(\vec{k})\longrightarrow -i 2 \pi k_y \ft{A} = \fft{\partial_y A(\vec{r})}
  \f]
  \param[in] a Fourier transform of a scalar field A
  \param[out] da Fourier transform of x-derivative of A
 */
void A_k2dya_k(double *a, double *da);

/*!
  \brief Compute z-derivative of scalar field (in reciprocal space)
  \details \f[
  \ft{A}(\vec{k})\longrightarrow -i 2 \pi k_z \ft{A}(\vec{k}) = \fft{\partial_z A(\vec{r})}
  \f]
  \param[in] a Fourier transform of a scalar field A
  \param[out] da Fourier transform of x-derivative of A
 */
void A_k2dza_k(double *a, double *da);

/*!
  \brief Compute reduced vorticity field from full vorticity field (reciprocal space)
  \details \f[
  \ft{\vec{\omega}}(\vec{k})\longrightarrow \ft{\vec{\zeta}}(\vec{k}) = \ft{\vec{\omega}}^*(\vec{k})
  \f]
  \param[in] omega full vorticity field (reciprocal space)
  \param[out] zetak reduced vorticity field (reciprocal space)
 */
void Omega_k2zeta_k(double **omega, double **zetak);

/*!
  \brief Compute contravariant reduced vorticity field from full vorticity field
  (reciprocal space)
  \details \f[
  \ft{\omega}^\alpha(\vec{k})\longrightarrow
  \ft{\zeta}^{\alpha}(\vec{k}) = \ft{\omega}^{\alpha*}(\vec{k})
  \f]
  \param[in] omega contravariant vorticity field (reciprocal space)
  \param[out] zetak contravariant reduced vorticity field (reciprocal space)
 */
void Omega_k2zeta_k_OBL(double **omega, double **zetak);

/*!
  \brief Compute reduced vorticity field from velocity field (reciprocal space)
  \details \f[
  \ft{\vec{u}}(\vec{k})\longrightarrow \ft{\vec{\zeta}}(\vec{k})
  \f]
  \param[in] u velocity field (reciprocal space)
  \param[out] zeta reduced vorticity field (reciprocal space)
  \param[out] uk_dc zero-wavenumber Fourier transform of u
 */
void U_k2zeta_k(double **u, double **zeta, double uk_dc[DIM]);

/*!
  \brief Compute solenoidal velocity field from reduced vorticity field (reciprocal space)
  \details \f[
  \ft{\vec{\zeta}}(\vec{k})\longrightarrow \ft{\vec{\omega}}(\vec{k})
  \underset{\vec{k}\cdot\ft{\vec{u}}=0}{\longrightarrow} \ft{\vec{u}}(\vec{k})
  \f]
  \param[in] zeta reduced vorticity field (reciprocal space)
  \param[in] uk_dc zero-wavenumber Fourier transform of the velocity field
  \param[out] u velocity field (reciprocal space)
 */
void Zeta_k2u_k(double **zeta, double uk_dc[DIM], double **u);

/*!
  \brief Compute contravariant vorticity field from reduced vorticity
  field (reciprocal space)
  \details \f[
  \ft{\zeta}^\alpha(\vec{k}) \longrightarrow
  \ft{\omega}^\alpha(\vec{k})
  \f]
  \param[in] zeta contravariant reduced vorticity field
  \param[out] omega contravariant vorticity field
 */
void Zeta_k2omega_k_OBL(double **zeta, double **omega);

/*!
  \brief Compute contravariant vorticity field from contravariant
  velocity field (reciprocal space)
  \details \f[
  \ft{u}^\alpha(\vec{k}) \longrightarrow \ft{\omega}^\alpha(\vec{k})
  \f]
  \param[in] u contravariant velocity field (reciprocal space)
  \param[out] omega contravariant vorticity field (reciprocal space)
  \param[out] uk_dc zero-wavenumbe Fourier transfrom of the
  contravariant velocity field
 */
void U_k2omega_k_OBL(double **u, double **omega, double uk_dc[DIM]);

/*!
  \brief Compute contravariant reduced vorticity field from contravariant
  velocity field (reciprocal space)
  \details \f[
  \ft{u}^\alpha(\vec{k}) \longrightarrow \ft{\zeta}^\alpha(\vec{k})
  \f]
  \param[in] u contravariant velocity field (reciprocal space)
  \param[out] zeta contravariant reduced vorticity field (reciprocal
  space)
  \param[out] uk_dc zero-wavenumber Fourier transform of the
  contravariant velocity field
 */
void U_k2zeta_k_OBL(double **u, double **zeta, double uk_dc[DIM]);

/*!
  \brief Compute contravariant colenoidal velocity field from
  contravariant vorticity field (reciprocal space)
  \details \f[
  \ft{\omega}^\alpha(\vec{k}) \longrightarrow \ft{u}^{\alpha}(\vec{k})
  \f]
  \param[in] omega contravariant vorticity field (reciprocal space)
  \param[in] uk_dc zero_wavenumber Fourier transform of the
  contravariant velocity field
  \param[out] u contravariant velocity field (reciprocal space)
 */
void Omega_k2u_k_OBL(double **omega, double uk_dc[DIM], double **u);
/*!
  \brief Compute contravariant solenoidal velocity field from
  contravariant reduced vorticity
  field (reciprocal space)
  \details \f[
  \ft{\zeta}^\alpha(\vec{k})\longrightarrow
  \ft{\omega}^\alpha(\vec{k}) \longrightarrow
  \ft{\omega}_\alpha(\vec{k}) \propto \epsilon_{\alpha\beta\gamma}k^{\beta}\ft{u}^{\gamma}
  \underset{k_\alpha \ft{u}^\alpha = 0}{\longrightarrow} \ft{u}^{\alpha}(\vec{k})
  \f]
  \param[in] zeta contravariant reduced vorticity field (reciprocal
  space)
  \param[in] uk_dc zero-wavenumber Fourier transform of the
  contravariant velocity field
  \param[out] u contravariant velocity field (reciprocal space)
 */
void Zeta_k2u_k_OBL(double **zeta, double uk_dc[DIM], double **u);

/*!
  Compute stress tensor
 */
void U_k2Stress_k(double **u, double *stress_k[QDIM]);

/*!
  Compute contravaraint stress tensor
 */
void U_k2Stress_k_OBL(double **u, double *stress_k[QDIM]);

/*!
  \brief Compute divergence of vector field (in reciprocal space)
  \details \f[
  \ft{\vec{u}}(\vec{k})\longrightarrow -i 2\pi\vec{k}\cdot\ft{\vec{u}} =
  \fft{\nabla\cdot \vec{u}(\vec{r})}
  \f]
  \param[in] u Fourier transform of vector field u
  \param[out] div Fourier transform of divergence of u
 */
void U_k2divergence_k(double **u, double *div);

/*!
  \brief Compute the curl of vector field (in reciprocal space)
  \details \f[
  \ft{u}(\vec{k})\longrightarrow -i 2\pi\vec{k}\times\ft{\vec{u}}(\vec{k}) = \fft{\nabla_{\vec{r}}\vec{u}(\vec{r})}
  \f]
  \param[in,out] u Fourier transform of vector field u (in), Fourier transform of curl of u (out)
 */
void U_k2rotation_k(double **u);

inline void orth2obl(const int &j, const int &i, int &i_oblique, int &i_oblique_plus, double &alpha, double &beta) {
    int    delta_j = j - NY / 2;
    double sign    = (degree_oblique >= 0.0 ? (double)delta_j : -(double)delta_j);

    if (!(delta_j == 0)) sign = sign / fabs(sign);

    i_oblique = (int)(sign * degree_oblique * delta_j) * sign;
    alpha     = (degree_oblique * delta_j - i_oblique) * sign;
    beta      = 1.0 - alpha;

    i_oblique      = (int)fmod(i + i_oblique + 4.0 * NX, NX);
    i_oblique_plus = (int)fmod(i_oblique + sign + 4.0 * NX, NX);
}
inline void obl2orth(const int &j, const int &i, int &i_plus, int &i_oblique, double &alpha, double &beta) {
    int    delta_j = j - NY / 2;
    double sign    = (degree_oblique >= 0.0 ? (double)delta_j : -(double)delta_j);

    if (!(delta_j == 0)) sign = sign / fabs(sign);

    i_oblique = (int)(sign * degree_oblique * delta_j) * sign + sign;
    alpha     = (i_oblique - degree_oblique * delta_j) * sign;
    beta      = 1.0 - alpha;

    i_oblique = (int)fmod(i + i_oblique + 4.0 * NX, NX);
    i_plus    = (int)fmod(i + sign + 2.0 * NX, NX);
}

/*!
  \brief Transform scalar field from rectangular to oblique coordinates
  \param[in,out] phi scalar (density) field to transform
 */
inline void phi2phi_oblique(double *phi) {
    int im;
    int im_ob;
    int im_ob_p;

    Copy_v1(work_v1, phi);

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

                phi[im] = (beta * work_v1[im_ob] + alpha * work_v1[im_ob_p]);
            }
        }
    }
}

/*!
  \brief Transform scalar field from oblique to rectangular coordinates
  \param[in,out] phi scalar (density) field to transform
 */
inline void phi_oblique2phi(double *phi) {
    int im;
    int im_ob;
    int im_p;

    Copy_v1(work_v1, phi);

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

                phi[im_ob] = beta * work_v1[im] + alpha * work_v1[im_p];
            }
        }
    }
}

// Allocate / Deallocate interpolation memory
inline void Init_Transform_obl() {
    if (SW_OBL_INT == spline_int) {
        int nthreads;
#ifndef _OPENMP
        nthreads = 1;
#else
#pragma omp parallel
        { nthreads = omp_get_num_threads(); }
#endif
        uspline       = (double ***)malloc(sizeof(double **) * nthreads);
        splineOblique = new splineSystem *[nthreads];

        for (int np = 0; np < nthreads; np++) {
            splineInit(splineOblique[np], NX, DX);
            uspline[np] = (double **)malloc(sizeof(double *) * DIM);

            for (int d = 0; d < DIM; d++) uspline[np][d] = alloc_1d_double(NX);
        }
    }
}

inline void Free_Transform_obl() {
    if (SW_OBL_INT == spline_int) {
        int nthreads;
#ifndef _OPENMP
        nthreads = 1;
#else
#pragma omp parallel
        { nthreads = omp_get_num_threads(); }
#endif
        for (int np = 0; np < nthreads; np++) {
            for (int d = 0; d < DIM; d++) free_1d_double(uspline[np][d]);

            free(uspline[np]);
            splineFree(splineOblique[np]);
        }
        delete[] splineOblique;
        free(uspline);
    }
}

// Periodic spline interpolation
inline void Spline_u_oblique_transform(double **uu, const OBL_TRANSFORM &flag) {
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
                us0[0][i] = uu[0][im] - sign * degree_oblique * uu[1][im];
                us0[1][i] = uu[1][im];
                us0[2][i] = uu[2][im];
            }  // i

            // compute interpolated points
            for (int d = DIM - 1; d >= 0; d--) {
                splineCompute(spl, us0[d]);

                for (int i = 0; i < NX; i++) {  // transformed coord
                    im    = (i * NY * NZ_) + (j * NZ_) + k;
                    dmy_x = fmod(i * DX + sign * degree_oblique * delta_y + 4.0 * LX, LX);  // original coord

                    uu[d][im] = splineFx(spl, dmy_x);
                    if (d == 0 && sign < 0) uu[d][im] += Shear_rate_eff * delta_y;
                }  // i
            }      // d

        }  // k
    }      // j
}

inline void U2u_oblique(double **uu) {
    int im;
    int im_ob;
    int im_ob_p;

    Copy_v3(work_v3, uu);

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

                // orthogonal grid -> oblique grid
                for (int d = 0; d < DIM; d++) {
                    uu[d][im] = beta * work_v3[d][im_ob] + alpha * work_v3[d][im_ob_p];
                }

                // orthogonal coordinates -> oblique coordinates
                // warning: mean shear flow is not removed
                uu[0][im] -= (degree_oblique * uu[1][im]);
            }
        }
    }
}

inline void U_oblique2u(double **uu, const bool &add_mean_flow = true) {
    int im;
    int im_ob;
    int im_p;

    Copy_v3(work_v3, uu);

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

                // oblique grid -> orthogonal grid
                for (int d = 0; d < DIM; d++) {
                    uu[d][im_ob] = beta * work_v3[d][im] + alpha * work_v3[d][im_p];
                }

                // oblique coordinates -> orthogonal coordinates
                // warning: mean shear flow is added by default!
                uu[0][im_ob] += (degree_oblique * uu[1][im_ob]);
                if (add_mean_flow) uu[0][im_ob] += Shear_rate_eff * (j - NY / 2.0);
            }
        }
    }
}

// contravariant stress tensor from oblique to orthogonal
inline void Stress_oblique2Stress(double **EE, const bool &add_mean_flow = true) {
    int     im;
    int     im_ob;
    int     im_p;
    double *work_v5[QDIM] = {work_v3[0], work_v3[1], work_v3[2], work_v2[0], work_v2[1]};
    Copy_v5(work_v5, EE);

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

                // oblique grid -> orthogonal grid
                for (int d = 0; d < QDIM; d++) {
                    EE[d][im_ob] = beta * work_v5[d][im] + alpha * work_v5[d][im_p];
                }

                // oblique coordinates -> orthogonal coordinates
                // warning: mean shear flow is added by default!
                {
                    // xx
                    EE[0][im_ob] += (2.0 * degree_oblique * EE[1][im_ob] + SQ(degree_oblique) * EE[3][im_ob]);
                    // xy
                    EE[1][im_ob] += (degree_oblique * EE[3][im_ob]);
                    // xz
                    EE[2][im_ob] += (degree_oblique * EE[4][im_ob]);
                }
                if (add_mean_flow) EE[1][im_ob] += (ETA * Shear_rate_eff);

            }  // k
        }      // j
    }          // i
}

inline void Transform_obl_u(double **uu, const OBL_TRANSFORM &flag) {
    if (SW_OBL_INT == linear_int) {
        if (flag == oblique2cartesian) {
            U_oblique2u(uu);
        } else if (flag == cartesian2oblique) {
            U2u_oblique(uu);
        } else {
            exit_job(EXIT_FAILURE);
        }
    } else if (SW_OBL_INT == spline_int) {
        Spline_u_oblique_transform(uu, flag);
    } else {
        exit_job(EXIT_FAILURE);
    }
}

inline void contra2co(double **contra) {
    int im;

    Copy_v3_k(work_v3, contra);

#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ_; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;

                contra[0][im] = work_v3[0][im] + degree_oblique * work_v3[1][im];
                contra[1][im] =
                    degree_oblique * work_v3[0][im] + (1. + degree_oblique * degree_oblique) * work_v3[1][im];
                contra[2][im] = work_v3[2][im];
            }
        }
    }
}

inline void co2contra(double **contra) {
    int im;

    Copy_v3_k(work_v3, contra);

#pragma omp parallel for private(im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ_; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;

                contra[0][im] =
                    (1. + degree_oblique * degree_oblique) * work_v3[0][im] - degree_oblique * work_v3[1][im];
                contra[1][im] = -degree_oblique * work_v3[0][im] + work_v3[1][im];
                contra[2][im] = work_v3[2][im];
            }
        }
    }
}

inline void contra2co_single(double contra[]) {
    double dmy[DIM];

    for (int d = 0; d < DIM; d++) {
        dmy[d] = contra[d];
    }

    contra[0] = dmy[0] + degree_oblique * dmy[1];
    contra[1] = degree_oblique * dmy[0] + (1. + degree_oblique * degree_oblique) * dmy[1];
    contra[2] = dmy[2];
}

inline void co2contra_single(double co[]) {
    double dmy[DIM];

    for (int d = 0; d < DIM; d++) {
        dmy[d] = co[d];
    }

    co[0] = (1. + degree_oblique * degree_oblique) * dmy[0] - degree_oblique * dmy[1];
    co[1] = -degree_oblique * dmy[0] + dmy[1];
    co[2] = dmy[2];
}

/*!
  \brief Compute Fourier transform of scalar field (in place)
  \details \f[A(\vec{r}) \longrightarrow \ft{A}(\vec{k})\f]
  \param[in,out] a scalar field A (input), Fourier transform of A (ouput)
 */

inline void A2a_k(double *a) {
#ifdef _FFT_IMKL
    long status = DftiComputeForward(imkl_p_fw, a);
#elif _FFT_FFTW
    fftw_execute_dft_r2c(fftw_p_fw, a, reinterpret_cast<fftw_complex *>(a));
#else
    initview_3d_double(NX, NY, NZ_, a, ooura_p.a);
    rdft3d(NX, NY, NZ, 1, ooura_p.a, ooura_p.t, ooura_p.ip, ooura_p.w);
    rdft3dsort(NX, NY, NZ, 1, ooura_p.a);
    Complex *ak = reinterpret_cast<Complex *>(a);
#pragma omp parallel for
    for (int i = 0; i < NX * NY * HNZ_; i++) ak[i] = std::conj(ak[i]);
#endif
}

/*!
  \brief Compute inverse Fourier transform of scalar field (in place)
  \details \f[\ft{A}(\vec{k}) \longrightarrow A(\vec{r})\f]
  \param[in,out] a Fourier transform of scalar field A (input), A (ouput)
 */

inline void A_k2a(double *a) {
#ifdef _FFT_IMKL
    long status = DftiComputeBackward(imkl_p_bw, a);
#elif _FFT_FFTW
    static const double scale = 1.0 / (NX * NY * NZ);
    Complex *           ak    = reinterpret_cast<Complex *>(a);
#pragma omp parallel for
    for (int i = 0; i < NX * NY * HNZ_; i++) ak[i] *= scale;
    fftw_execute_dft_c2r(fftw_p_bw, reinterpret_cast<fftw_complex *>(a), a);
#else
    static const double scale = 2.0 / (NX * NY * NZ);
    Complex *           ak    = reinterpret_cast<Complex *>(a);
#pragma omp parallel for
    for (int i = 0; i < NX * NY * HNZ_; i++) ak[i] = scale * std::conj(ak[i]);

    initview_3d_double(NX, NY, NZ_, a, ooura_p.a);
    rdft3dsort(NX, NY, NZ, -1, ooura_p.a);
    rdft3d(NX, NY, NZ, -1, ooura_p.a, ooura_p.t, ooura_p.ip, ooura_p.w);
#endif
}

/*!
  \brief Compute inverse Fourier transform of scalar field
  \details \f[\ft{A}(\vec{k}) \longrightarrow A(\vec{r})\f]
  \param[in] a_k Fourier transform of scalar field A
  \param[out] a_x A in real space
 */
inline void A_k2a_out(double *a_k, double *a_x) {
    Copy_v1_k(a_x, a_k);
    A_k2a(a_x);
}

/*!
  \brief Compute Fourier transform of scalar field
  \details \f[A(\vec{r}) \longrightarrow \ft{A}(\vec{k})\f]
  \param[in] a_x Real-space scalar field A
  \param[out] a_k Fourier transform of A
 */
inline void A2a_k_out(double *a_x, double *a_k) {
    Copy_v1(a_k, a_x);
    A2a_k(a_k);
}

/*!
  \brief Compute (real space) gradient of scalar field (in reciprocal space)
  \details \f[
  \ft{A}(\vec{k})\longrightarrow -i 2\pi\vec{k}\ft{A}(\vec{k}) = \fft{\nabla_{\vec{r}} A(\vec{r})}
  \f]
  \param[in] a Fourier transform of scalar field A
  \param[out] da Fourier transform of gradient of A
 */
inline void A_k2da_k(double *a, double **da) {
    A_k2dxa_k(a, da[0]);
    A_k2dya_k(a, da[1]);
    A_k2dza_k(a, da[2]);
}

inline int Calc_KX_Ooura(const int &i, const int &j, const int &k) { return (i > HNX) ? i - NX : i; }
inline int Calc_KY_Ooura(const int &i, const int &j, const int &k) {
    assert(i < NX);
    assert(j < NY);
    return (j > HNY) ? j - NY : j;
}
inline int Calc_KZ_Ooura(const int &i, const int &j, const int &k) {
    assert(i < NX);
    assert(j < NY);
    return k / 2;
}
inline void Truncate_general(double *a, const Index_range &ijk_range) {
    int im;
#pragma omp parallel for private(im)
    for (int i = ijk_range.istart; i <= ijk_range.iend; i++) {
        for (int j = ijk_range.jstart; j <= ijk_range.jend; j++) {
            for (int k = ijk_range.kstart; k <= ijk_range.kend; k++) {
                assert((abs(Calc_KY_Ooura(i, j, k)) >= TRN_Y || abs(Calc_KX_Ooura(i, j, k)) >= TRN_X ||
                        Calc_KZ_Ooura(i, j, k) >= TRN_Z));
                im    = (i * NY * NZ_) + (j * NZ_) + k;
                a[im] = 0.0;
            }
        }
    }
}

/*!
  \brief Orzag's 2/3 rule to de-alias Fourier Transform
  \details Supresses the high wavenumbers according to Orzag's rule.
  Eliminates  aliasing of the non-linear quadratic terms (i.e advection).
  See Ch. 11.5 of Boyd's book for more detailes (available online
  <a href="http://www-personal.umich.edu/~jpboyd/BOOK_Spectral2000.html">here
  </a>).
  \param[in,out] a Fourier Transform of field to dealias
 */
inline void Truncate_two_third_rule_ooura(double *a) {
    static Index_range dmy_range;
    const int          trn_z2 = 2 * TRN_Z;
    {
        dmy_range.istart = 0;
        dmy_range.iend   = NX - 1;
        dmy_range.jstart = 0;
        dmy_range.jend   = NY - 1;
        dmy_range.kstart = trn_z2;
        dmy_range.kend   = NZ_ - 1;
        Truncate_general(a, dmy_range);
    }

    {
        dmy_range.istart = 0;
        dmy_range.iend   = NX - 1;
        dmy_range.jstart = TRN_Y;
        dmy_range.jend   = NY - TRN_Y;
        dmy_range.kstart = 0;
        dmy_range.kend   = trn_z2 - 1;
        Truncate_general(a, dmy_range);
    }

    {
        dmy_range.istart = TRN_X;
        dmy_range.iend   = NX - TRN_X;
        dmy_range.jstart = 0;
        dmy_range.jend   = TRN_Y - 1;
        dmy_range.kstart = 0;
        dmy_range.kend   = trn_z2 - 1;
        Truncate_general(a, dmy_range);
    }

    {
        dmy_range.istart = TRN_X;
        dmy_range.iend   = NX - TRN_X;
        dmy_range.jstart = NY - TRN_Y + 1;
        dmy_range.jend   = NY - 1;
        dmy_range.kstart = 0;
        dmy_range.kend   = trn_z2 - 1;
        Truncate_general(a, dmy_range);
    }
}
#endif
