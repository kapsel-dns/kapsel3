/*!
  \file fft_wrapper.cxx
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief FFT wrapper routines for reciprocal space calculations
 */

#include "fft_wrapper.h"

// FFTW
#ifdef _FFT_IMKL
DFTI_DESCRIPTOR_HANDLE imkl_p_fw, imkl_p_bw;
#elif _FFT_FFTW
fftw_plan fftw_p_fw, fftw_p_bw;
#else
ooura_plan ooura_p;
#endif

double **ucp;
double * phi, *phi_p, *phi_sum, **up, **u, *rhop, *phi_wall;
double **work_v3, **work_v2, *work_v1;

int *   KX_int, *KY_int, *KZ_int;
double *K2, *IK2;

splineSystem **splineOblique;
double ***     uspline;

Index_range *ijk_range_two_third_filter;
int          n_ijk_range_two_third_filter;

int (*Calc_KX)(const int &i, const int &j, const int &k);
int (*Calc_KY)(const int &i, const int &j, const int &k);
int (*Calc_KZ)(const int &i, const int &j, const int &k);
void (*Truncate_two_third_rule)(double *a);

inline void Free_K(void) {
    free_1d_double(IK2);
    free_1d_double(K2);
    free_1d_int(KZ_int);
    free_1d_int(KY_int);
    free_1d_int(KX_int);
}
inline void Init_K(void) {
    KX_int = alloc_1d_int(NX * NY * NZ_);
    KY_int = alloc_1d_int(NX * NY * NZ_);
    KZ_int = alloc_1d_int(NX * NY * NZ_);
    K2     = alloc_1d_double(NX * NY * NZ_);
    IK2    = alloc_1d_double(NX * NY * NZ_);

    int kx, ky, kz;
    int im;
#pragma omp parallel for private(kx, ky, kz, im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ_; k++) {
                im = (i * NY * NZ_) + (j * NZ_) + k;
                kx = Calc_KX(i, j, k);
                ky = Calc_KY(i, j, k);
                kz = Calc_KZ(i, j, k);

                KX_int[im] = kx;
                KY_int[im] = ky;
                KZ_int[im] = kz;

                K2[im] = SQ(WAVE_X * (double)kx) + SQ(WAVE_Y * (double)ky) + SQ(WAVE_Z * (double)kz);
                if (K2[im] > 0.0) {
                    IK2[im] = 1. / K2[im];
                } else {
                    IK2[im] = 0.0;
                }
            }
        }
    }
}

inline void Init_fft_ooura(void) {
    int n, nt;
    nt            = MAX(NX, NY);
    n             = MAX(nt, NZ / 2);
    ooura_p.ip    = alloc_1d_int(2 + (int)sqrt((double)n + 0.5));
    ooura_p.t     = alloc_1d_double(8 * nt);
    ooura_p.w     = alloc_1d_double(n / 2 + NZ / 4);
    ooura_p.ip[0] = 0;
    ooura_p.a     = allocview_3d_double(NX, NY, NZ_);
}
inline void Free_fft_ooura(void) {
    free_1d_double(ooura_p.w);
    free_1d_double(ooura_p.t);
    free_1d_int(ooura_p.ip);
    freeview_3d_double(ooura_p.a);
}

#ifdef _FFT_FFTW
inline void Init_fft_fftw(void) {
#ifdef _OPENMP
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
    double *      input  = alloc_1d_double(NX * NY * NZ_);
    fftw_complex *output = reinterpret_cast<fftw_complex *>(input);
    {
        fftw_iodim64 dims[DIM] = {
            {.n = NX, .is = NY * NZ_, .os = NY * HNZ_}, {.n = NY, .is = NZ_, .os = HNZ_}, {.n = NZ, .is = 1, .os = 1}};
        fftw_p_fw = fftw_plan_guru64_dft_r2c(DIM, dims, 0, (fftwf_iodim64 *)0, input, output, FFTW_MEASURE);
    }
    {
        fftw_iodim64 dims[DIM] = {
            {.n = NX, .is = NY * HNZ_, .os = NY * NZ_}, {.n = NY, .is = HNZ_, .os = NZ_}, {.n = NZ, .is = 1, .os = 1}};
        fftw_p_bw = fftw_plan_guru64_dft_c2r(DIM, dims, 0, (fftwf_iodim64 *)0, output, input, FFTW_MEASURE);
    }
    free_1d_double(input);
}
inline void Free_fft_fftw(void) {
    fftw_destroy_plan(fftw_p_fw);
    fftw_destroy_plan(fftw_p_bw);

#ifdef _OPENMP
    fftw_cleanup_threads();
#else
    fftw_cleanup();
#endif
}
#endif

#ifdef _FFT_IMKL
inline void Init_fft_imkl(void) {
    long m, n, k, status;
    long lengths[DIM]         = {NX, NY, NZ};
    long strides_in[DIM + 1]  = {0, NY * NZ_, NZ_, 1};
    long strides_out[DIM + 1] = {0, NY * HNZ_, HNZ_, 1};
    status                    = DftiCreateDescriptor(&imkl_p_fw, DFTI_DOUBLE, DFTI_REAL, DIM, lengths);
    status                    = DftiSetValue(imkl_p_fw, DFTI_PLACEMENT, DFTI_INPLACE);
    status                    = DftiSetValue(imkl_p_fw, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status                    = DftiSetValue(imkl_p_fw, DFTI_INPUT_STRIDES, strides_in);
    status                    = DftiSetValue(imkl_p_fw, DFTI_OUTPUT_STRIDES, strides_out);
    status                    = DftiCommitDescriptor(imkl_p_fw);

    status = DftiCopyDescriptor(imkl_p_fw, &imkl_p_bw);
    status = DftiSetValue(imkl_p_bw, DFTI_INPUT_STRIDES, strides_out);
    status = DftiSetValue(imkl_p_bw, DFTI_OUTPUT_STRIDES, strides_in);
    status = DftiSetValue(imkl_p_bw, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(NX * NY * NZ));
    status = DftiCommitDescriptor(imkl_p_bw);
}
inline void Free_fft_imkl(void) {
    long status;
    status = DftiFreeDescriptor(&imkl_p_fw);
    status = DftiFreeDescriptor(&imkl_p_bw);
}
#endif

void Init_fft(void) {
#ifdef _FFT_IMKL
    fprintf(stderr, "# Intel Math Kernel Library FFT is selected.\n");
    Init_fft_imkl();
#elif _FFT_FFTW
    fprintf(stderr, "# FFTW is selected.\n");
    Init_fft_fftw();
#else
    fprintf(stderr, "# Ooura rdft3d is selected.\n");
    Init_fft_ooura();
#endif
    {
        Calc_KX = Calc_KX_Ooura;
        Calc_KY = Calc_KY_Ooura;
        Calc_KZ = Calc_KZ_Ooura;

        Init_K();

        Truncate_two_third_rule = Truncate_two_third_rule_ooura;
        {  // for 2/3 rule
            Index_range dmy_range[] = {
                {0, TRN_X - 1, 0, TRN_Y - 1, 0, 2 * TRN_Z - 1},
                {0, TRN_X - 1, NY - TRN_Y + 1, NY - 1, 0, 2 * TRN_Z - 1},
                {NX - TRN_X + 1, NX - 1, 0, TRN_Y - 1, 0, 2 * TRN_Z - 1},
                {NX - TRN_X + 1, NX - 1, NY - TRN_Y + 1, NY - 1, 0, 2 * TRN_Z - 1},
            };
            n_ijk_range_two_third_filter = sizeof(dmy_range) / sizeof(Index_range);
            ijk_range_two_third_filter   = new Index_range[n_ijk_range_two_third_filter];
            for (int n = 0; n < n_ijk_range_two_third_filter; n++) {
                ijk_range_two_third_filter[n] = dmy_range[n];
            }
        }
    }
}

void Free_fft(void) {
#ifdef _FFT_IMKL
    Free_fft_imkl();
#elif _FFT_FFTW
    Free_fft_fftw();
#else
    Free_fft_ooura();
#endif
    Free_K();
    delete[] ijk_range_two_third_filter;
}

inline void A_k2dja_k_primitive(double *      a,
                                double *      da,
                                const int &   nx,
                                const int &   ny,
                                const int &   hnz_,
                                const int *   k_int,
                                const double &wave_base) {
    int    k2;
    int    im;
    double wavenumber;
#pragma omp parallel for private(k2, wavenumber, im)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < hnz_; k++) {
                k2         = 2 * k;
                im         = (i * NY * NZ_) + (j * NZ_) + k2;
                wavenumber = k_int[im] * wave_base;
                da[im]     = -wavenumber * a[im + 1];
                da[im + 1] = wavenumber * a[im];
            }
        }
    }
}
void A_k2dxa_k(double *a, double *da) { A_k2dja_k_primitive(a, da, NX, NY, HNZ_, KX_int, WAVE_X); }
void A_k2dya_k(double *a, double *da) { A_k2dja_k_primitive(a, da, NX, NY, HNZ_, KY_int, WAVE_Y); }
void A_k2dza_k(double *a, double *da) { A_k2dja_k_primitive(a, da, NX, NY, HNZ_, KZ_int, WAVE_Z); }

void Omega_k2zeta_k(double **omega, double **zetak) {
    int im;
    {
#pragma omp parallel for private(im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < NZ_; k++) {
                    im = (i * NY * NZ_) + (j * NZ_) + k;
                    /*
                    if(KX_int[im] != 0){
                      zetak[0][im] = omega[1][im];
                      zetak[1][im] = omega[2][im];
                    }else if(KZ_int[im] != 0){
                      zetak[0][im] = omega[1][im];
                      zetak[1][im] = omega[0][im];
                    }else{
                      zetak[0][im] = omega[0][im];
                      zetak[1][im] = omega[2][im];
                    }
                    */
                    if (KX_int[im] != 0) {
                        zetak[0][im] = omega[1][im];
                        zetak[1][im] = omega[2][im];
                    } else if (KY_int[im] != 0) {
                        zetak[0][im] = omega[2][im];
                        zetak[1][im] = omega[0][im];
                    } else {
                        zetak[0][im] = omega[0][im];
                        zetak[1][im] = omega[1][im];
                    }
                }
            }
        }
    }
    assert(zetak[0][0] == 0.0);
    assert(zetak[1][0] == 0.0);
}

void Omega_k2zeta_k_OBL(double **omega, double **zetak) {
    int im;
    {
#pragma omp parallel for private(im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < NZ_; k++) {
                    im = (i * NY * NZ_) + (j * NZ_) + k;
                    if (KX_int[im] != 0) {
                        zetak[0][im] = omega[1][im];
                        zetak[1][im] = omega[2][im];
                    } else if (KZ_int[im] != 0) {
                        zetak[0][im] = omega[1][im];
                        zetak[1][im] = omega[0][im];
                    } else {
                        zetak[0][im] = omega[0][im];
                        zetak[1][im] = omega[2][im];
                    }
                }
            }
        }
    }
    assert(zetak[0][0] == 0.0);
    assert(zetak[1][0] == 0.0);
}

void U_k2Stress_k(double **u, double *stress_k[QDIM]) {
    // Stress_k は 5成分
    double dmy[DIM] = {0., 0., 0.};
    int    k2;
    int    im0;
    int    im1;
    double ks[DIM];
    double u_dmy[DIM][2];

    // save uk_dc
    for (int d = 0; d < DIM; d++) {
        dmy[d]  = u[d][0];
        u[d][0] = 0.0;
    }

#pragma omp parallel for private(im0, im1, k2, ks, u_dmy)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < HNZ_; k++) {
                k2    = 2 * k;
                im0   = (i * NY * NZ_) + (j * NZ_) + k2;
                im1   = (i * NY * NZ_) + (j * NZ_) + k2 + 1;
                ks[0] = KX_int[im0] * WAVE_X;
                ks[1] = KY_int[im0] * WAVE_Y;
                ks[2] = KZ_int[im0] * WAVE_Z;
                for (int d = 0; d < DIM; d++) {
                    u_dmy[d][0] = ETA * u[d][im0];
                    u_dmy[d][1] = ETA * u[d][im1];
                }

                stress_k[0][im0] = -(ks[0] * u_dmy[0][1] * 2.);
                stress_k[0][im1] = (ks[0] * u_dmy[0][0] * 2.);

                stress_k[1][im0] = -(ks[0] * u_dmy[1][1] + ks[1] * u_dmy[0][1]);
                stress_k[1][im1] = (ks[0] * u_dmy[1][0] + ks[1] * u_dmy[0][0]);

                stress_k[2][im0] = -(ks[0] * u_dmy[2][1] + ks[2] * u_dmy[0][1]);
                stress_k[2][im1] = (ks[0] * u_dmy[2][0] + ks[2] * u_dmy[0][0]);

                stress_k[3][im0] = -(ks[1] * u_dmy[1][1] * 2.);
                stress_k[3][im1] = (ks[1] * u_dmy[1][0] * 2.);

                stress_k[4][im0] = -(ks[1] * u_dmy[2][1] + ks[2] * u_dmy[1][1]);
                stress_k[4][im1] = (ks[1] * u_dmy[2][0] + ks[2] * u_dmy[1][0]);
            }
        }
    }

    // reset uk_dc
    for (int d = 0; d < DIM; d++) {
        u[d][0] = dmy[d];
    }
}

void U_k2Stress_k_OBL(double **u, double *stress_k[QDIM]) {
    // Stress_k は 5成分
    // E^{\mu\mu}
    double dmy[DIM] = {0., 0., 0.};
    int    k2;
    int    im0;
    int    im1;
    double ks[DIM];
    double u_dmy[DIM][2];

    // save uk_dc
    for (int d = 0; d < DIM; d++) {
        dmy[d]  = u[d][0];
        u[d][0] = 0.0;
    }

#pragma omp parallel for private(im0, im1, k2, ks, u_dmy)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < HNZ_; k++) {
                k2    = 2 * k;
                im0   = (i * NY * NZ_) + (j * NZ_) + k2;
                im1   = im0 + 1;
                ks[0] = KX_int[im0] * WAVE_X;
                ks[1] = KY_int[im0] * WAVE_Y;
                ks[2] = KZ_int[im0] * WAVE_Z;
                co2contra_single(ks);  // contra

                for (int d = 0; d < DIM; d++) {
                    u_dmy[d][0] = ETA * u[d][im0];
                    u_dmy[d][1] = ETA * u[d][im1];
                }  // contra

                // E^{xx}
                stress_k[0][im0] = -(ks[0] * u_dmy[0][1] * 2.);
                stress_k[0][im1] = (ks[0] * u_dmy[0][0] * 2.);

                // E^{xy}
                stress_k[1][im0] = -(ks[0] * u_dmy[1][1] + ks[1] * u_dmy[0][1]);
                stress_k[1][im1] = (ks[0] * u_dmy[1][0] + ks[1] * u_dmy[0][0]);

                // E^{xz}
                stress_k[2][im0] = -(ks[0] * u_dmy[2][1] + ks[2] * u_dmy[0][1]);
                stress_k[2][im1] = (ks[0] * u_dmy[2][0] + ks[2] * u_dmy[0][0]);

                // E^{yy}
                stress_k[3][im0] = -(ks[1] * u_dmy[1][1] * 2.);
                stress_k[3][im1] = (ks[1] * u_dmy[1][0] * 2.);

                // E^{yz}
                stress_k[4][im0] = -(ks[1] * u_dmy[2][1] + ks[2] * u_dmy[1][1]);
                stress_k[4][im1] = (ks[1] * u_dmy[2][0] + ks[2] * u_dmy[1][0]);
            }
        }
    }

    // reset uk_dc
    for (int d = 0; d < DIM; d++) {
        u[d][0] = dmy[d];
    }
}

void U_k2zeta_k(double **u, double **zeta, double uk_dc[DIM]) {
    double ks[DIM];
    double omega_re[DIM];
    double omega_im[DIM];
    int    k2;
    int    im;
    {
        uk_dc[0] = u[0][0];
        uk_dc[1] = u[1][0];
        uk_dc[2] = u[2][0];
#pragma omp parallel for private(ks, omega_re, omega_im, k2, im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2          = 2 * k;
                    im          = (i * NY * NZ_) + (j * NZ_) + k2;
                    ks[0]       = KX_int[im] * WAVE_X;
                    ks[1]       = KY_int[im] * WAVE_Y;
                    ks[2]       = KZ_int[im] * WAVE_Z;
                    omega_re[0] = -(ks[1] * u[2][im + 1] - ks[2] * u[1][im + 1]);
                    omega_im[0] = (ks[1] * u[2][im] - ks[2] * u[1][im]);

                    omega_re[1] = -(ks[2] * u[0][im + 1] - ks[0] * u[2][im + 1]);
                    omega_im[1] = (ks[2] * u[0][im] - ks[0] * u[2][im]);

                    omega_re[2] = -(ks[0] * u[1][im + 1] - ks[1] * u[0][im + 1]);
                    omega_im[2] = (ks[0] * u[1][im] - ks[1] * u[0][im]);

                    if (KX_int[im] != 0) {
                        zeta[0][im]     = omega_re[1];
                        zeta[0][im + 1] = omega_im[1];
                        zeta[1][im]     = omega_re[2];
                        zeta[1][im + 1] = omega_im[2];
                    } else if (KY_int[im] != 0) {
                        zeta[0][im]     = omega_re[2];
                        zeta[0][im + 1] = omega_im[2];
                        zeta[1][im]     = omega_re[0];
                        zeta[1][im + 1] = omega_im[0];
                    } else {
                        zeta[0][im]     = omega_re[0];
                        zeta[0][im + 1] = omega_im[0];
                        zeta[1][im]     = omega_re[1];
                        zeta[1][im + 1] = omega_im[1];
                    }

                    /*
                    if(KX_int[im] != 0){
                        zeta[0][im] = omega_re[1];
                        zeta[0][im + 1] = omega_im[1];
                        zeta[1][im] = omega_re[2];
                        zeta[1][im + 1] = omega_im[2];
                    }else if(KZ_int[im] != 0){
                        zeta[0][im] = omega_re[1];
                        zeta[0][im + 1] = omega_im[1];
                        zeta[1][im] = omega_re[0];
                        zeta[1][im + 1] = omega_im[0];
                    }else{
                        zeta[0][im] = omega_re[0];
                        zeta[0][im + 1] = omega_im[0];
                        zeta[1][im] = omega_re[2];
                        zeta[1][im + 1] = omega_im[2];
                    }
                    */
                }
            }
        }
    }
    assert(zeta[0][0] == 0.0);
    assert(zeta[1][0] == 0.0);
}

void U_k2omega_k_OBL(double **u, double **omega, double uk_dc[DIM]) {
    double u_re[DIM];
    double u_im[DIM];
    double ks[DIM];
    int    k2;
    int    im;

    uk_dc[0] = u[0][0];
    uk_dc[1] = u[1][0];
    uk_dc[2] = u[2][0];
    {
#pragma omp parallel for private(u_re, u_im, ks, k2, im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2    = 2 * k;
                    im    = (i * NY * NZ_) + (j * NZ_) + k2;
                    ks[0] = KX_int[im] * WAVE_X;
                    ks[1] = KY_int[im] * WAVE_Y;
                    ks[2] = KZ_int[im] * WAVE_Z;

                    u_re[0] = u[0][im];
                    u_im[0] = u[0][im + 1];
                    u_re[1] = u[1][im];
                    u_im[1] = u[1][im + 1];
                    u_re[2] = u[2][im];
                    u_im[2] = u[2][im + 1];
                    contra2co_single(u_re);
                    contra2co_single(u_im);

                    omega[0][im]     = -(ks[1] * u_im[2] - ks[2] * u_im[1]);
                    omega[0][im + 1] = (ks[1] * u_re[2] - ks[2] * u_re[1]);

                    omega[1][im]     = -(ks[2] * u_im[0] - ks[0] * u_im[2]);
                    omega[1][im + 1] = (ks[2] * u_re[0] - ks[0] * u_re[2]);

                    omega[2][im]     = -(ks[0] * u_im[1] - ks[1] * u_im[0]);
                    omega[2][im + 1] = (ks[0] * u_re[1] - ks[1] * u_re[0]);

                }  // k
            }      // j
        }          // i
    }
    assert(omega[0][0] == 0.0);
    assert(omega[1][0] == 0.0);
    assert(omega[2][0] == 0.0);
}

void U_k2zeta_k_OBL(double **u, double **zeta, double uk_dc[DIM]) {
    double ks[DIM];
    double u_re[DIM];
    double u_im[DIM];
    double omega_re[DIM];
    double omega_im[DIM];
    int    k2;
    int    im;

    uk_dc[0] = u[0][0];
    uk_dc[1] = u[1][0];
    uk_dc[2] = u[2][0];
    {
#pragma omp parallel for private(ks, u_re, u_im, omega_re, omega_im, k2, im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2    = 2 * k;
                    im    = (i * NY * NZ_) + (j * NZ_) + k2;
                    ks[0] = KX_int[im] * WAVE_X;
                    ks[1] = KY_int[im] * WAVE_Y;
                    ks[2] = KZ_int[im] * WAVE_Z;

                    u_re[0] = u[0][im];
                    u_im[0] = u[0][im + 1];
                    u_re[1] = u[1][im];
                    u_im[1] = u[1][im + 1];
                    u_re[2] = u[2][im];
                    u_im[2] = u[2][im + 1];
                    contra2co_single(u_re);
                    contra2co_single(u_im);

                    omega_re[0] = -(ks[1] * u_im[2] - ks[2] * u_im[1]);
                    omega_im[0] = (ks[1] * u_re[2] - ks[2] * u_re[1]);

                    omega_re[1] = -(ks[2] * u_im[0] - ks[0] * u_im[2]);
                    omega_im[1] = (ks[2] * u_re[0] - ks[0] * u_re[2]);

                    omega_re[2] = -(ks[0] * u_im[1] - ks[1] * u_im[0]);
                    omega_im[2] = (ks[0] * u_re[1] - ks[1] * u_re[0]);

                    if (KX_int[im] != 0) {
                        zeta[0][im]     = omega_re[1];
                        zeta[0][im + 1] = omega_im[1];
                        zeta[1][im]     = omega_re[2];
                        zeta[1][im + 1] = omega_im[2];
                    } else if (KZ_int[im] != 0) {
                        zeta[0][im]     = omega_re[1];
                        zeta[0][im + 1] = omega_im[1];
                        zeta[1][im]     = omega_re[0];
                        zeta[1][im + 1] = omega_im[0];
                    } else {
                        zeta[0][im]     = omega_re[0];
                        zeta[0][im + 1] = omega_im[0];
                        zeta[1][im]     = omega_re[2];
                        zeta[1][im + 1] = omega_im[2];
                    }
                }
            }
        }
    }
    assert(zeta[0][0] == 0.0);
    assert(zeta[1][0] == 0.0);
}

void Zeta_k2u_k(double **zeta, double uk_dc[DIM], double **u) {
    double omega_re[DIM];
    double omega_im[DIM];
    double dmy1_re, dmy2_re;
    double dmy1_im, dmy2_im;
    double kx;
    double ky;
    double kz;
    double ik2;
    int    im;
    int    k2;
    double dmy;

    {
#pragma omp parallel for private(omega_re, omega_im, dmy1_re, dmy2_re, dmy1_im, dmy2_im, kx, ky, kz, ik2, im, k2, dmy)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2  = 2 * k;
                    im  = (i * NY * NZ_) + (j * NZ_) + k2;
                    kx  = WAVE_X * KX_int[im];
                    ky  = WAVE_Y * KY_int[im];
                    kz  = WAVE_Z * KZ_int[im];
                    ik2 = IK2[im];

                    dmy1_re = zeta[0][im];
                    dmy1_im = zeta[0][im + 1];
                    dmy2_re = zeta[1][im];
                    dmy2_im = zeta[1][im + 1];

                    if (KX_int[im] != 0) {
                        omega_re[0] = -(1. / kx) * (ky * dmy1_re + kz * dmy2_re);
                        omega_im[0] = -(1. / kx) * (ky * dmy1_im + kz * dmy2_im);
                        omega_re[1] = dmy1_re;
                        omega_im[1] = dmy1_im;
                        omega_re[2] = dmy2_re;
                        omega_im[2] = dmy2_im;
                    } else if (KY_int[im] != 0) {
                        dmy         = -(kz / ky);
                        omega_re[0] = dmy2_re;
                        omega_im[0] = dmy2_im;
                        omega_re[1] = dmy * dmy1_re;
                        omega_im[1] = dmy * dmy1_im;
                        omega_re[2] = dmy1_re;
                        omega_im[2] = dmy1_im;
                    } else {
                        omega_re[0] = dmy1_re;
                        omega_im[0] = dmy1_im;
                        omega_re[1] = dmy2_re;
                        omega_im[1] = dmy2_im;
                        omega_re[2] = 0.0;
                        omega_im[2] = 0.0;
                    }

                    /*
                    if(KX_int[im] != 0){
                        omega_re[0] = -(1./kx)*(ky * dmy1_re + kz * dmy2_re);
                        omega_im[0] = -(1./kx)*(ky * dmy1_im + kz * dmy2_im);
                        omega_re[1] = dmy1_re;
                        omega_im[1] = dmy1_im;
                        omega_re[2] = dmy2_re;
                        omega_im[2] = dmy2_im;


                    }else if(KZ_int[im] != 0){
                        dmy = -(ky/kz);
                        omega_re[0] = dmy2_re;
                        omega_im[0] = dmy2_im;
                        omega_re[1] = dmy1_re;
                        omega_im[1] = dmy1_im;
                        omega_re[2] = dmy*dmy1_re;
                        omega_im[2] = dmy*dmy1_im;
                    }else{
                        omega_re[0] = dmy1_re;
                        omega_im[0] = dmy1_im;
                        omega_re[1] = 0.0;
                        omega_im[1] = 0.0;
                        omega_re[2] = dmy2_re;
                        omega_im[2] = dmy2_im;
                    }
                    */
                    kx *= ik2;
                    ky *= ik2;
                    kz *= ik2;
                    u[0][im]     = -(ky * omega_im[2] - kz * omega_im[1]);
                    u[0][im + 1] = (ky * omega_re[2] - kz * omega_re[1]);
                    u[1][im]     = -(kz * omega_im[0] - kx * omega_im[2]);
                    u[1][im + 1] = (kz * omega_re[0] - kx * omega_re[2]);
                    u[2][im]     = -(kx * omega_im[1] - ky * omega_im[0]);
                    u[2][im + 1] = (kx * omega_re[1] - ky * omega_re[0]);
                }
            }
        }
        u[0][0] = uk_dc[0];
        u[1][0] = uk_dc[1];
        u[2][0] = uk_dc[2];
    }
}

void Zeta_k2u_k_OBL(double **zeta, double uk_dc[DIM], double **u) {
    double omega_re[DIM];
    double omega_im[DIM];
    double dmy1_re, dmy2_re;
    double dmy1_im, dmy2_im;
    double kx;
    double ky;
    double kz;
    double ik2;
    int    im;
    int    k2;
    double dmy;

    {
#pragma omp parallel for private(omega_re, omega_im, dmy1_re, dmy2_re, dmy1_im, dmy2_im, kx, ky, kz, ik2, im, k2, dmy)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2  = 2 * k;
                    im  = (i * NY * NZ_) + (j * NZ_) + k2;
                    kx  = WAVE_X * KX_int[im];
                    ky  = WAVE_Y * KY_int[im];
                    kz  = WAVE_Z * KZ_int[im];
                    ik2 = IK2[im];

                    dmy1_re = zeta[0][im];
                    dmy1_im = zeta[0][im + 1];
                    dmy2_re = zeta[1][im];
                    dmy2_im = zeta[1][im + 1];

                    if (KX_int[im] != 0) {
                        omega_re[0] = -(1. / kx) * (ky * dmy1_re + kz * dmy2_re);
                        omega_im[0] = -(1. / kx) * (ky * dmy1_im + kz * dmy2_im);
                        omega_re[1] = dmy1_re;
                        omega_im[1] = dmy1_im;
                        omega_re[2] = dmy2_re;
                        omega_im[2] = dmy2_im;

                    } else if (KZ_int[im] != 0) {
                        dmy         = -(ky / kz);
                        omega_re[0] = dmy2_re;
                        omega_im[0] = dmy2_im;
                        omega_re[1] = dmy1_re;
                        omega_im[1] = dmy1_im;
                        omega_re[2] = dmy * dmy1_re;
                        omega_im[2] = dmy * dmy1_im;
                    } else {
                        omega_re[0] = dmy1_re;
                        omega_im[0] = dmy1_im;
                        omega_re[1] = 0.0;
                        omega_im[1] = 0.0;
                        omega_re[2] = dmy2_re;
                        omega_im[2] = dmy2_im;
                    }

                    // From contravariant to covariant
                    contra2co_single(omega_re);
                    contra2co_single(omega_im);

                    kx *= ik2;
                    ky *= ik2;
                    kz *= ik2;

                    u[0][im]     = -(ky * omega_im[2] - kz * omega_im[1]);  // contra
                    u[0][im + 1] = (ky * omega_re[2] - kz * omega_re[1]);   // contra

                    u[1][im]     = -(kz * omega_im[0] - kx * omega_im[2]);  // contra
                    u[1][im + 1] = (kz * omega_re[0] - kx * omega_re[2]);   // contra

                    u[2][im]     = -(kx * omega_im[1] - ky * omega_im[0]);  // contra
                    u[2][im + 1] = (kx * omega_re[1] - ky * omega_re[0]);   // contra
                }
            }
        }
        u[0][0] = uk_dc[0];
        u[1][0] = uk_dc[1];
        u[2][0] = uk_dc[2];
    }
}

void Omega_k2u_k_OBL(double **omega, double uk_dc[DIM], double **u) {
    double omega_re[DIM];
    double omega_im[DIM];
    double kx;
    double ky;
    double kz;
    double ik2;
    int    k2;
    int    im;

#pragma omp parallel for private(omega_re, omega_im, kx, ky, kz, ik2, k2, im)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < HNZ_; k++) {
                k2 = 2 * k;
                im = (i * NY * NZ_) + (j * NZ_) + k2;

                ik2 = IK2[im];
                kx  = WAVE_X * KX_int[im] * ik2;
                ky  = WAVE_Y * KY_int[im] * ik2;
                kz  = WAVE_Z * KZ_int[im] * ik2;

                omega_re[0] = omega[0][im];
                omega_im[0] = omega[0][im + 1];
                omega_re[1] = omega[1][im];
                omega_im[1] = omega[1][im + 1];
                omega_re[2] = omega[2][im];
                omega_im[2] = omega[2][im + 1];

                // From contravariant to covariant
                contra2co_single(omega_re);
                contra2co_single(omega_im);

                u[0][im]     = -(ky * omega_im[2] - kz * omega_im[1]);  // contra
                u[0][im + 1] = (ky * omega_re[2] - kz * omega_re[1]);   // contra

                u[1][im]     = -(kz * omega_im[0] - kx * omega_im[2]);  // contra
                u[1][im + 1] = (kz * omega_re[0] - kx * omega_re[2]);   // contra

                u[2][im]     = -(kx * omega_im[1] - ky * omega_im[0]);  // contra
                u[2][im + 1] = (kx * omega_re[1] - ky * omega_re[0]);   // contra
            }                                                           // k
        }                                                               // j
    }                                                                   // i

    u[0][0] = uk_dc[0];
    u[1][0] = uk_dc[1];
    u[2][0] = uk_dc[2];
}

void Zeta_k2omega_k_OBL(double **zeta, double **omega) {
    double dmy1, dmy2;
    double kx;
    double ky;
    double kz;
    int    im;
    double dmy;

    {
#pragma omp parallel for private(dmy1, dmy2, kx, ky, kz, im, dmy)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < NZ_; k++) {
                    im = (i * NY * NZ_) + (j * NZ_) + k;
                    kx = WAVE_X * KX_int[im];
                    ky = WAVE_Y * KY_int[im];
                    kz = WAVE_Z * KZ_int[im];

                    dmy1 = zeta[0][im];
                    dmy2 = zeta[1][im];

                    if (KX_int[im] != 0) {
                        omega[0][im] = -(1. / kx) * (ky * dmy1 + kz * dmy2);
                        omega[1][im] = dmy1;
                        omega[2][im] = dmy2;

                    } else if (KZ_int[im] != 0) {
                        dmy          = -(ky / kz);
                        omega[0][im] = dmy2;
                        omega[1][im] = dmy1;
                        omega[2][im] = dmy * dmy1;

                    } else {
                        omega[0][im] = dmy1;
                        omega[1][im] = 0.0;
                        omega[2][im] = dmy2;
                    }
                }
            }
        }

        for (int d = 0; d < DIM; d++) {
            omega[d][0] = 0.0;
        }
    }
}

void U_k2divergence_k(double **u, double *div) {
    double ks[DIM];
    int    k2;
    int    im0, im1;
    {
#pragma omp parallel for private(ks, k2, im0, im1)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2  = 2 * k;
                    im0 = (i * NY * NZ_) + (j * NZ_) + k2;
                    im1 = im0 + 1;

                    ks[0] = KX_int[im0] * WAVE_X;
                    ks[1] = KY_int[im0] * WAVE_Y;
                    ks[2] = KZ_int[im0] * WAVE_Z;
                    // div[im0] = 0.0;
                    // div[im1] = 0.0;
                    div[im0] = -(ks[0] * u[0][im1] + ks[1] * u[1][im1] + ks[2] * u[2][im1]);
                    div[im1] = (ks[0] * u[0][im0] + ks[1] * u[1][im0] + ks[2] * u[2][im0]);
                }
            }
        }
    }
}

void U_k2rotation_k(double **u) {
    double ks[DIM];
    double dmy_u_re[DIM];
    double dmy_u_im[DIM];
    double dmy_rot_re[DIM];
    double dmy_rot_im[DIM];

    int k2;
    int im;
    {
#pragma omp parallel for private(ks, dmy_u_re, dmy_u_im, dmy_rot_re, dmy_rot_im, k2, im)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for (int k = 0; k < HNZ_; k++) {
                    k2 = 2 * k;
                    im = (i * NY * NZ_) + (j * NZ_) + k2;

                    ks[0] = KX_int[im] * WAVE_X;
                    ks[1] = KY_int[im] * WAVE_Y;
                    ks[2] = KZ_int[im] * WAVE_Z;

                    for (int d = 0; d < DIM; d++) {
                        dmy_u_re[d] = u[d][im];
                        dmy_u_im[d] = u[d][im + 1];
                    }

                    dmy_rot_re[0] = -(ks[1] * dmy_u_im[2] - ks[2] * dmy_u_im[1]);
                    dmy_rot_im[0] = (ks[1] * dmy_u_re[2] - ks[2] * dmy_u_re[1]);

                    dmy_rot_re[1] = -(ks[2] * dmy_u_im[0] - ks[0] * dmy_u_im[2]);
                    dmy_rot_im[1] = (ks[2] * dmy_u_re[0] - ks[0] * dmy_u_re[2]);

                    dmy_rot_re[2] = -(ks[0] * dmy_u_im[1] - ks[1] * dmy_u_im[0]);
                    dmy_rot_im[2] = (ks[0] * dmy_u_re[1] - ks[1] * dmy_u_re[0]);

                    for (int d = 0; d < DIM; d++) {
                        u[d][im]     = dmy_rot_re[d];
                        u[d][im + 1] = dmy_rot_im[d];
                    }
                }
            }
        }
    }
}
