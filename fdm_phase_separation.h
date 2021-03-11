#ifndef FDM_PHASE_SEPARATION_H
#define FDM_PHASE_SEPARATION_H

#include <hdf5.h>
#include <hdf5_hl.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "H5Cpp.h"
#include "fdm.h"
#include "fft_wrapper.h"
#include "input.h"

extern double * cp;
extern double * psi;
extern double * psi_o;
extern double * psicp;
extern double * psicp_o;
extern double * phi_obl;
extern double **stress;
extern double **stress_o;

inline double potential_deriv(double x) {
    double retval;
    if (SW_POTENTIAL == Landau) {
        retval = gl.a * (x * x * x) - gl.b * x;
    } else if (SW_POTENTIAL == Flory_Huggins) {
        retval = (1. / fh.na) - (1. / fh.nb) + (log(x) / fh.na) - (log(1. - x) / fh.nb) + fh.chi * (1. - 2. * x);
    }
    return retval;
}

void Calc_cp(double *phi, double *psi, double *cp);
void Calc_cp_wall(double *phi, double *phi_p, double *phi_wall, double *psi, double *cp);
void Calc_cp_OBL(double *phi, double *psi, double *cp, const double degree_oblique);

void Cp2stress(double *cp, double *psi, double **stress);
void Cp2stress_OBL(double *cp, double *psi, double **stress, const double degree_oblique);

void Set_poisson_rhs_ps(double **u, double **adv, double **lap, double **stress, double *s, CTime &jikan);
void Set_poisson_rhs_viscosity(double **u,
                               double **u_s,
                               double **adv,
                               double **lap,
                               double **stress,
                               double * eta_s,
                               double * s,
                               CTime &  jikan);
void Set_poisson_rhs_viscosity_OBL(double **u,
                                   double **u_s,
                                   double **adv,
                                   double **lap,
                                   double **stress,
                                   double * eta_s,
                                   double * s,
                                   CTime &  jikan);

void Update_u_stress_euler(double **u, double **stress, CTime &jikan);
void Update_u_stress_ab2(double **u, double **stress, double **stress_o, CTime &jikan);

void Update_psi_euler(double *psi, double **u, double *phi, double *cp, CTime &jikan);
void Update_psi_euler_OBL(double *psi, double **u, double *cp, CTime &jikan, const double degree_oblique);

void Init_phase_separation(double *phi, double *psi);

void Output_xdmf_sca(std::string filename, std::string hdffilename, std::string dataname, CTime &jikan);
void Output_xdmf_vec(std::string filename,
                     std::string hdffilename,
                     std::string dataname_0,
                     std::string dataname_1,
                     std::string dataname_2,
                     CTime &     jikan);
void Output_xdmf_particle(std::string filename, std::string hdffilename, CTime &jikan);
void Output_xdmf_particle_single(std::string filename, std::string hdffilename, CTime &jikan);

int Output_hdf5_sca(std::string filename, std::string dataname, double *data_1d, int count);
int Output_hdf5_vec(std::string filename,
                    std::string dataname_0,
                    std::string dataname_1,
                    std::string dataname_2,
                    double **   data_3d,
                    int         count);
int Output_hdf5_particle(std::string filename, Particle *p, int count);
int Output_hdf5_particle_single(std::string filename, Particle *p, int count);

void Psi2eta(double *psi, double *eta);

void xdmf_output(CTime jikan);
#endif