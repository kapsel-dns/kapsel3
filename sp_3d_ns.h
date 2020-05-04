/*!
  \file sp_3d_ns.h
  \author Y. Nakayama
  \date 2006/11/30
  \version 1.9
  \brief Main program file (header)
  \todo documentation
 */
#ifndef SP_3D_NS_H
#define SP_3D_NS_H

#define NDEBUG
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstring>

#include "aux_field.h"
#include "ewald_wrapper.h"
#include "f_particle.h"
#include "fdm.h"
#include "fluid_solver.h"
#include "init_fluid.h"
#include "init_particle.h"
#include "input.h"
#include "macro.h"
#include "make_phi.h"
#include "md_force.h"
#include "operate_electrolyte.h"
#include "operate_omega.h"
#include "operate_surface.h"
#include "output.h"
#include "particle_solver.h"
#include "resume.h"
#include "variable.h"
#include "wall.h"

#ifdef _FFT_IMKL
#include <mkl.h>
#endif

enum Count_SW { INIT, ADD, MEAN, SNAP_MEAN, SHOW };

inline void Electrolyte_free_energy(const Count_SW &OPERATION,
                                    FILE *          fout,
                                    Particle *      p,
                                    double **       Concentration_rhs1,
                                    const CTime &   jikan) {
    static const char *labels[] = {"", "nu", "radius", "xi", "M", "I", "kBT", "kBT/M"};
    static char        line_label[1 << 10];
    static const char *labels_free_energy_nosalt[] = {
        "", "total", "ideal_gas", "electrostatic", "liquid_charge", "total_counterion"};
    static const char *labels_free_energy_salt[] = {
        "", "total", "ideal_gas", "electrostatic", "liquid_charge", "total_positive_ion", "total_negative_ion"};
    // static char line_label_free_energy[1<<10];

    if (OPERATION == INIT) {
        {
            sprintf(line_label, "#");
            for (int d = 1; d < sizeof(labels) / sizeof(char *); d++) {
                sprintf(line_label, "%s%d:%s ", line_label, d, labels[d]);
            }
        }
    } else if (OPERATION == SHOW || OPERATION == MEAN) {
        if (OPERATION == MEAN) {
            fprintf(fout, "%s\n", line_label);
            fprintf(fout,
                    "%g %g %g %g %g %g %g\n",
                    NU,
                    A * DX,
                    XI * DX,
                    MASS[p[0].spec],
                    MOI[p[0].spec],
                    kBT,
                    kBT * IMASS[p[0].spec]);
        }
        {
            double free_energy[3];
            Calc_free_energy_PB(Concentration_rhs1, p, free_energy, up[0], up[1], up[2], jikan);
            double  ion_density = 0.;
            double *n_solute    = new double[N_spec];
            Count_solute_each(n_solute, Concentration_rhs1, p, phi, up[0]);
            for (int n = 0; n < N_spec; n++) {
                ion_density += n_solute[n] * Valency_e[n];
            }
            char line0[1 << 10];
            char line1[1 << 10];
            int  d;
            int  dstart;
            {
                if (OPERATION == SHOW) {
                    dstart = 1;
                    sprintf(line0, "#%d:%s ", dstart, "time");
                    sprintf(line1, "%d ", jikan.ts);
                } else {
                    dstart = 0;
                    sprintf(line0, "#");
                    sprintf(line1, "");
                }
            }
            if (N_spec == 1) {
                for (d = 1; d < sizeof(labels_free_energy_nosalt) / sizeof(char *); d++) {
                    sprintf(line0, "%s%d:%s ", line0, dstart + d, labels_free_energy_nosalt[d]);
                }
                sprintf(line1,
                        "%s%.15g %.15g %.15g %.15g %.15g",
                        line1,
                        free_energy[0],
                        free_energy[1],
                        free_energy[2],
                        ion_density,
                        n_solute[0]);
                sprintf(line0, "%s\n", line0);
                sprintf(line1, "%s\n", line1);
            } else {
                for (d = 1; d < sizeof(labels_free_energy_salt) / sizeof(char *); d++) {
                    sprintf(line0, "%s%d:%s ", line0, dstart + d, labels_free_energy_salt[d]);
                }
                sprintf(line1,
                        "%s%.15g %.15g %.15g %.15g %.15g %.15g",
                        line1,
                        free_energy[0],
                        free_energy[1],
                        free_energy[2],
                        ion_density,
                        n_solute[0],
                        n_solute[1]);
                sprintf(line0, "%s\n", line0);
                sprintf(line1, "%s\n", line1);
            }
            fprintf(fout, "%s%s", line0, line1);
            delete[] n_solute;
        }
    } else {
        fprintf(stderr, "invalid OPERATION in Electrolyte_free_energy().\n");
        exit_job(EXIT_FAILURE);
    }
}
inline double Calc_instantaneous_shear_rate(double **zeta, double uk_dc[DIM], double **u  // working memory
) {
    static const double hivolume  = Ivolume * POW3(DX) * 2.;
    static int          ny0       = NY / 4;
    static int          ny1       = 3 * NY / 4;
    double              srate_eff = 0.0;

    Zeta_k2u_k(zeta, uk_dc, u);
    A_k2dya_k(u[0], u[1]);
    A_k2a(u[1]);
    {
#pragma omp parallel for reduction(+ : srate_eff)
        for (int i = 0; i < NX; i++) {
            for (int j = ny0; j < ny1; j++) {
                for (int k = 0; k < NZ; k++) {
                    srate_eff += u[1][(i * NY * NZ_) + (j * NZ_) + k];
                }
            }
        }
    }
    return (srate_eff * hivolume);
}

inline double Calc_local_gradient_y_OBL(const double *field, int im) {
	const double INV_2DX = 1. / (2. * DX);
    int          i, j, k;
	im2ijk(im, &i, &j, &k);
	// periodic boundary condition
	int ip1, jp1;
	int im1, jm1;
	ip1 = adj(1, i, NX);
	jp1 = adj(1, j, NY);
	im1 = adj(-1, i, NX);
	jm1 = adj(-1, j, NY);
	// set adjacent meshes
	int im_ip1_jp1 = ijk2im(ip1, jp1, k);
	int im_im1_jp1 = ijk2im(im1, jp1, k);
	int im_ip1_jm1 = ijk2im(ip1, jm1, k);
	int im_im1_jm1 = ijk2im(im1, jm1, k);
	// interior division
	double field_p1 = 0.5 * ((1 - degree_oblique) * field[im_ip1_jp1] + (1 + degree_oblique) * field[im_im1_jp1]);
	double field_m1 = 0.5 * ((1 - degree_oblique) * field[im_im1_jm1] + (1 + degree_oblique) * field[im_ip1_jm1]);
	double local_dfield_dy = (field_p1 - field_m1) * INV_2DX;
	return local_dfield_dy;
}

inline void Calc_shear_rate_eff() {
    int                 im;
    static const double ivolume    = Ivolume * POW3(DX);
    double              s_rate_eff = 0.;
#pragma omp parallel for reduction(+ : s_rate_eff)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im = i * NY * NZ_ + j * NZ_ + k;
				s_rate_eff += Calc_local_gradient_y_OBL(u[0], im);
			}
		}
	}
	s_rate_eff *= ivolume;
	Shear_rate_eff += s_rate_eff;
}

inline double Update_strain(double &     shear_strain_realized,
                            const CTime &jikan,
                            double **    zeta,
                            double       uk_dc[DIM],
                            double **    u  // working memory
) {
    double srate_eff = -Calc_instantaneous_shear_rate(zeta, uk_dc, u);
    shear_strain_realized += srate_eff * jikan.dt_fluid;
    return srate_eff;
}

inline void Calc_fluid_stress(double **u, double *eta, double &fluid_stress) {
    int                 im;
    double              dux_dx           = 0.;
    double              dux_dy           = 0.;
    double              shear_rate_local = 0.;
    static const double ivolume          = Ivolume * POW3(DX);
#pragma omp parallel for reduction(+ : fluid_stress)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im               = i * NY * NZ_ + j * NZ_ + k;
                shear_rate_local = Shear_rate_eff + Calc_local_gradient_y_OBL(u[0], im);
                fluid_stress += shear_rate_local * eta[im];
            }
        }
    }
    fluid_stress *= ivolume;
}

inline void Calc_interfacial_stress(double *psi, double &interfacial_stress) {
    int                 im;
    double              dpsi_dx = 0.;
    double              dpsi_dy = 0.;
    static const double ivolume = Ivolume * POW3(DX);
#pragma omp parallel for reduction(+ : interfacial_stress)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                im      = i * NY * NZ_ + j * NZ_ + k;
				dpsi_dx = calc_gradient_o1_to_o1(psi, im, 0);
          		dpsi_dy = Calc_local_gradient_y_OBL(psi, im);
          		interfacial_stress += dpsi_dx * dpsi_dy;
           	}
	    }	
    }
    interfacial_stress *= -ivolume * ps.alpha;
}

inline void Mean_shear_stress(const Count_SW &OPERATION,
                              FILE *          fout,
                              Particle *      p,
                              const CTime &   jikan,
                              const double &  srate_eff) {
    static const char *labels_zz_dc[] = {
        "", "time", "shear_rate_temporal", "shear_strain_temporal", "shear_stress_temporal", "viscosity"};
    static const char *labels_zz_ac[] = {"",
                                         "time",
                                         "shear_rate_temporal",
                                         "shear_strain_temporal",
                                         "shear_stress_temporal",
                                         "shear_inertia_stress_temporal",
                                         "apparent_shear_stress"};
    static const char *labels_le[]    = {"",
                                      "time",
                                      "shear_rate",
                                      "degree_oblique",
                                      "shear_strain_temporal",
                                      "lj_dev_stress_temporal",
                                      "shear_stress_temporal_old",
                                      "shear_stress_temporal_new",
                                      "reynolds_stress",
                                      "fluid_stress",
                                      "interfacial_stress",
                                      "apparent_stress",
                                      "viscosity"};

    static char line_label[1 << 10];

    if (OPERATION == INIT) {
        sprintf(line_label, "#");
        if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM ||
            SW_EQ == Shear_NS_LE_CH_FDM) {
            for (int d = 1; d < sizeof(labels_le) / sizeof(char *); d++)
                sprintf(line_label, "%s%d:%s ", line_label, d, labels_le[d]);
        } else if (SW_EQ == Shear_Navier_Stokes) {
            if (!Shear_AC) {
                for (int d = 1; d < sizeof(labels_zz_dc) / sizeof(char *); d++)
                    sprintf(line_label, "%s%d:%s ", line_label, d, labels_zz_dc[d]);
            } else {
                for (int d = 1; d < sizeof(labels_zz_dc) / sizeof(char *); d++)
                    sprintf(line_label, "%s%d:%s ", line_label, d, labels_zz_ac[d]);
            }
        } else {
            fprintf(stderr, "Error: Incorrect Shear calculation\n");
            exit_job(EXIT_FAILURE);
        }
        fprintf(fout, "%s\n", line_label);
    } else if (OPERATION == SHOW) {
        double stress[DIM][DIM]           = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
        double hydro_stress[DIM][DIM]     = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
        double hydro_stress_new[DIM][DIM] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
        double fluid_stress               = 0.;
        double interfacial_stress         = 0.;
        double apparent_stress            = 0.;
        double strain_output              = Shear_strain_realized;
        if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM ||
            SW_EQ == Shear_NS_LE_CH_FDM) {
            Calc_hydro_stress(jikan, p, phi, Hydro_force, hydro_stress);
            Calc_hydro_stress(jikan, p, phi, Hydro_force_new, hydro_stress_new);
            double dev_stress     = (SW_PT == rigid ? rigid_dev_shear_stress_lj : dev_shear_stress_lj);
            double dev_stress_rot = (SW_PT == rigid ? rigid_dev_shear_stress_rot : dev_shear_stress_rot);
            double ETA_EFF        = ETA;
            fluid_stress          = ETA_EFF * srate_eff;
			if (PHASE_SEPARATION) Calc_interfacial_stress(psi, interfacial_stress);
            if (VISCOSITY_CHANGE) {
                // volume-averaged eta
                double dmy;
                if (SW_POTENTIAL == Landau) {
                    dmy = (1. + ps.ratio) / 2.;
                } else if (SW_POTENTIAL == Flory_Huggins) {
                    dmy = ps.ratio;
                }
                ETA_EFF = (ETA_A - ETA_B) * dmy + ETA_B;

				if (ETA_A != ETA_B) Calc_fluid_stress(u, eta_s, fluid_stress);
            }
            apparent_stress = hydro_stress_new[1][0] + Inertia_stress + dev_stress + fluid_stress + interfacial_stress;
            fprintf(fout,
                    "%16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g %16.8g\n",
                    jikan.time,
                    srate_eff,
                    degree_oblique,
                    strain_output,
                    dev_stress,
                    hydro_stress[1][0],
                    hydro_stress_new[1][0],
                    Inertia_stress,
                    fluid_stress,
                    interfacial_stress,
                    apparent_stress,
                    apparent_stress / srate_eff);
        } else if (SW_EQ == Shear_Navier_Stokes) {
            if (!Shear_AC) {
                Calc_shear_stress(jikan, p, phi, Shear_force, stress);
                fprintf(fout,
                        "%16.8g %16.8g %16.8g %16.8g %16.8g\n",
                        jikan.time,
                        srate_eff,
                        strain_output,
                        -stress[1][0],
                        -stress[1][0] / srate_eff);
            } else {
                Calc_shear_stress(jikan, p, phi, Shear_force, stress);
                fprintf(fout,
                        "%16.8g %16.8g %16.8g %16.8g %16.8g %16.8g\n",
                        jikan.time,
                        srate_eff,
                        strain_output,
                        -stress[1][0],
                        Inertia_stress,
                        -stress[1][0] + Inertia_stress);
            }
        }
    } else {
        fprintf(stderr, "invalid OPERATION in Mean_shear_stress().\n");
        exit_job(EXIT_FAILURE);
    }
}

#endif
