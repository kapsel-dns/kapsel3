#ifndef FDM_MATRIX_SOLVER_H
#define FDM_MATRIX_SOLVER_H

#include "fdm.h"
#include "fft_wrapper.h"
#include "input.h"

#ifdef _LIS_SOLVER
#include "lis.h"
#endif

#ifdef _LIS_SOLVER
extern LIS_INT *   ptr_ns, *idx_ns_csr;
extern LIS_SCALAR *val_ns_csr;
extern LIS_MATRIX  A_ns;
extern LIS_VECTOR  b_ns, x_ns;
extern LIS_SOLVER  lis_solver_ns;
extern LIS_INT     is_ns, ie_ns;

extern LIS_INT *   ptr_ch, *idx_ch_csr;
extern LIS_SCALAR *val_ch_csr;
extern LIS_MATRIX  A_ch;
extern LIS_VECTOR  b_ch, x_ch;
extern LIS_SOLVER  lis_solver_ch;
extern LIS_INT     is_ch, ie_ch;
#else
struct work_bicgstab {
    double  eps;
    int     maxiter;
    double *p;
    double *r;
    double *t;
    double *Ap;
    double *At;
    double *r0s;
    double *x;
    double *Ax;
};

extern work_bicgstab wm_ns;
extern work_bicgstab wm_ch;
#endif

#ifdef _LIS_SOLVER
void Init_lis(int argc, char *argv[]);
void Mem_alloc_lis(void);
void Free_lis(void);
#else
void        Init_ns(void);
void        Init_ch(void);
inline void Calc_Ax(double *x, int *idx, double *val, int *row_ptr, double *ans, int iend) {
#pragma omp parallel for
    for (int i = 0; i < iend; i++) {
        double sum = 0.;
        for (int ii = row_ptr[i]; ii < row_ptr[i + 1]; ii++) {
            sum += val[ii] * x[idx[ii]];
        }
        ans[i] = sum;
    }
}
void Mem_alloc_matrix_solver(void);
void Free_matrix_solver(void);
void bicgstab(int *idx, double *val, int *row_ptr, double *b, work_bicgstab &wm, int iend);
#endif

void NS_MAC_solver_implicit(double **u, double *pressure, double **u_s, const CTime &jikan, int is_ns, int ie_ns);
void CHNS_MAC_solver_implicit(double **    u,
                              double *     pressure,
                              double **    u_s,
                              double **    stress_s,
                              const CTime &jikan,
                              int          is_ns,
                              int          ie_ns);
void CHNS_MAC_solver_implicit_viscosity(double **    u,
                                        double *     pressure,
                                        double **    u_s,
                                        double **    stress_s,
                                        double *     eta_s,
                                        const CTime &jikan,
                                        int          is_ns,
                                        int          ie_ns);

void NS_MAC_solver_implicit_OBL(double **    u,
                                double *     pressure,
                                double **    u_s,
                                const CTime &jikan,
                                const double degree_oblique,
                                int          is_ns,
                                int          ie_ns);
void CHNS_MAC_solver_implicit_OBL(double **    u,
                                  double *     pressure,
                                  double **    u_s,
                                  double **    stress_s,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ns,
                                  int          ie_ns);
void CHNS_MAC_solver_implicit_viscosity_OBL(double **    u,
                                            double *     pressure,
                                            double **    u_s,
                                            double **    stress_s,
                                            double *     eta_s,
                                            const CTime &jikan,
                                            const double degree_oblique,
                                            int          is_ns,
                                            int          ie_ns);

void CH_solver_implicit_euler(double *     psi,
                              double *     psi_all,
                              double *     psi_o,
                              double *     phi,
                              double *     phi_p,
                              double *     phi_wall,
                              double *     phi_wall_prime,
                              double **    u,
                              const CTime &jikan,
                              int          is_ch,
                              int          ie_ch);

void CH_solver_implicit_bdfab(double *     psi,
                              double *     psi_all,
                              double *     psi_o,
                              double *     psi_all_o,
                              double *     phi,
                              double *     phi_p,
                              double *     phi_wall,
                              double *     phi_wall_prime,
                              double **    u,
                              const CTime &jikan,
                              int          is_ch,
                              int          ie_ch);

void CH_solver_implicit_euler_OBL(double *     psi,
                                  double *     psi_o,
                                  double *     phi,
                                  double **    u,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ch,
                                  int          ie_ch);
void CH_solver_implicit_bdfab_OBL(double *     psi,
                                  double *     psi_o,
                                  double *     phi,
                                  double **    u,
                                  const CTime &jikan,
                                  const double degree_oblique,
                                  int          is_ch,
                                  int          ie_ch);

void Make_potential_deriv(double *f_prime, double *psi);
#endif