/*!
  \file md_force.cxx
  \brief Routines to compute MD forces on particles
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
 */

#include "md_force.h"

double *Hydro_force;
double *Hydro_force_new;

#define Cell_length 16

void compute_particle_dipole_quincke(double *mu_space, const double *mu_body, quaternion &q) {
    double e_omega_space[DIM] = {0.0, 0.0, 0.0};
    rigid_body_rotation(e_omega_space, quincke.e_omega, q, BODY2SPACE);
    mu_space[0]            = quincke.n[1] * e_omega_space[2] - quincke.n[2] * e_omega_space[1];
    mu_space[1]            = quincke.n[2] * e_omega_space[0] - quincke.n[0] * e_omega_space[2];
    mu_space[2]            = quincke.n[0] * e_omega_space[1] - quincke.n[1] * e_omega_space[0];
    const double magnitude = mu_body[0] / sqrt(SQ(mu_space[0]) + SQ(mu_space[1]) + SQ(mu_space[2]));
    mu_space[0] *= magnitude;
    mu_space[1] *= magnitude;
    mu_space[2] *= magnitude;
    mu_space[2] -= magnitude * ewald_param.Pz_factor;
}
void compute_particle_dipole_quincke_image(double *mu_space, const double *mu_body, quaternion &q) {
    double e_omega_space[DIM] = {0.0, 0.0, 0.0};
    rigid_body_rotation(e_omega_space, quincke.e_omega, q, BODY2SPACE);
    mu_space[0]            = quincke.n[1] * e_omega_space[2] - quincke.n[2] * e_omega_space[1];
    mu_space[1]            = quincke.n[2] * e_omega_space[0] - quincke.n[0] * e_omega_space[2];
    mu_space[2]            = quincke.n[0] * e_omega_space[1] - quincke.n[1] * e_omega_space[0];
    const double magnitude = mu_body[0] / sqrt(SQ(mu_space[0]) + SQ(mu_space[1]) + SQ(mu_space[2]));
    mu_space[0] *= -magnitude;
    mu_space[1] *= -magnitude;
    mu_space[2] *= magnitude;
    mu_space[2] -= magnitude * ewald_param.Pz_factor;
}
void compute_particle_dipole_standard(double *mu_space, const double *mu_body, quaternion &q) {
    rigid_body_rotation(mu_space, mu_body, q, BODY2SPACE);
}
void (*compute_particle_dipole)(double *mu_space, const double *mu_body, quaternion &q);
void (*compute_particle_dipole_image)(double *mu_space, const double *mu_body, quaternion &q);

void Calc_f_Lennard_Jones_shear_cap_primitive_lnk(
    Particle *p,
    void (*distance0_func)(const double *x1, const double *x2, double &r12, double *x12),
    const double cap) {
    // Particle 変数の f に
    // !! +=
    //で足す. f の初期値 が正しいと仮定している!!
    const double pair_cutoff           = A_R_cutoff * LJ_dia;
    double       r_ij_vec[DIM]         = {0.0, 0.0, 0.0};
    double       r_ij                  = 0.0;
    double       shear_stress[2]       = {0.0, 0.0};
    double       rigid_shear_stress[2] = {0.0, 0.0};

    // List Constructor
    int  i, j;
    int *lscl;
    lscl = alloc_1d_int(Particle_Number);
    int    lc[DIM];
    double lc_r[DIM];
    int    mc[DIM];
    int    lcyz, lcxyz;
    int    ic[DIM], cn;
    lc[0] = int(NX / Cell_length);
    lc[1] = int(NY / Cell_length);
    lc[2] = int(NZ / Cell_length);
    for (int d = 0; d < DIM; d++) {
        lc_r[d] = Cell_length * DX;
    }
    lcyz  = lc[1] * lc[2];
    lcxyz = lc[0] * lcyz;

    int *head;
    head = alloc_1d_int(lcxyz);

#pragma omp parallel for
    for (int d = 0; d < lcxyz; d++) head[d] = -1;
    for (int n = 0; n < Particle_Number; n++) {
        Particle *p_n = &p[n];
        for (int d = 0; d < DIM; d++) {
            mc[d] = int((*p_n).x[d] / lc_r[d]);
        }
        cn       = mc[0] * lcyz + mc[1] * lc[2] + mc[2];
        lscl[n]  = head[cn];
        head[cn] = n;
    }

    for (ic[0] = 0; ic[0] < lc[0]; ic[0]++) {
        for (ic[1] = 0; ic[1] < lc[1]; ic[1]++) {
            for (ic[2] = 0; ic[2] < lc[2]; ic[2]++) {
                cn = ic[0] * lcyz + ic[1] * lc[2] + ic[2];
                // Scan the neighbor cells (including itself) of cell c
                int icl[DIM], icls[DIM];
                for (icl[0] = ic[0] - 1; icl[0] <= ic[0] + 1; icl[0]++) {
                    for (icl[1] = ic[1] - 1; icl[1] <= ic[1] + 1; icl[1]++) {
                        for (icl[2] = ic[2] - 1; icl[2] <= ic[2] + 1; icl[2]++) {
                            // Consider periodic condition
                            for (int d = 0; d < DIM; d++) {
                                icls[d] = icl[d];
                                if (icls[d] < 0) {
                                    icls[d] = lc[d] - 1;
                                }
                                if (icls[d] > (lc[d] - 1)) {
                                    icls[d] = 0;
                                }
                            }
                            // Calculate the scalar cell index of the neighbor cell
                            int cl = ((icls[0] + lc[0]) % lc[0]) * lcyz + ((icls[1] + lc[1]) % lc[1]) * lc[2] +
                                     ((icls[2] + lc[2]) % lc[2]);
                            // Scan atom i in cell c
                            i = head[cn];
                            while (i != -1) {
                                j             = head[cl];
                                int rigidID_i = -1;
                                if (SW_PT == rigid) rigidID_i = Particle_RigidID[i];

                                while (j != -1) {
                                    if (i > j && !rigid_chain(i, j) && !obstacle_chain(p[i].spec, p[j].spec)) {
                                        distance0_func(p[i].x, p[j].x, r_ij, r_ij_vec);

                                        if (r_ij < pair_cutoff) {
                                            double dmy_r = 0.0;

                                            dmy_r = MIN(cap / r_ij, Lennard_Jones_f(r_ij, LJ_dia, EPSILON, LJ_powers));

                                            {
                                                // spherical particle forces
                                                double dmy_fi[DIM] = {0.0, 0.0, 0.0};
                                                for (int d = 0; d < DIM; d++) {
                                                    dmy_fi[d] = (dmy_r) * (-r_ij_vec[d]);

                                                    p[i].fr[d] += dmy_fi[d];
                                                    p[j].fr[d] -= dmy_fi[d];
                                                }

                                                // stress
                                                shear_stress[0] += (dmy_fi[0] * r_ij_vec[1]);

                                                // rigid body forces & torques
                                                // particles treated as additive LJ centers: overlaps are not corrected
                                                if (SW_PT == rigid) {
                                                    int rigidID_j = Particle_RigidID[j];

                                                    for (int d = 0; d < DIM; d++) {
                                                        forceGrs[rigidID_i][d] += dmy_fi[d];
                                                        forceGrs[rigidID_j][d] -= dmy_fi[d];
                                                    }

                                                    torqueGrs[rigidID_i][0] +=
                                                        ((GRvecs[i][1] * dmy_fi[2] - GRvecs[i][2] * dmy_fi[1]));
                                                    torqueGrs[rigidID_i][1] +=
                                                        ((GRvecs[i][2] * dmy_fi[0] - GRvecs[i][0] * dmy_fi[2]));
                                                    torqueGrs[rigidID_i][2] +=
                                                        ((GRvecs[i][0] * dmy_fi[1] - GRvecs[i][1] * dmy_fi[0]));

                                                    torqueGrs[rigidID_j][0] +=
                                                        ((GRvecs[j][1] * dmy_fi[2] - GRvecs[j][2] * dmy_fi[1]));
                                                    torqueGrs[rigidID_j][1] +=
                                                        ((GRvecs[j][2] * dmy_fi[0] - GRvecs[j][0] * dmy_fi[2]));
                                                    torqueGrs[rigidID_j][2] +=
                                                        ((GRvecs[j][0] * dmy_fi[1] - GRvecs[j][1] * dmy_fi[0]));

                                                    double R_IJ_vec[DIM];
                                                    double R_IJ;
                                                    distance0_func(xGs[rigidID_i], xGs[rigidID_j], R_IJ, R_IJ_vec);
                                                    rigid_shear_stress[0] += (dmy_fi[0] * R_IJ_vec[1]);
                                                }
                                            }
                                        }
                                    }
                                    j = lscl[j];
                                }
                                i = lscl[i];
                            }
                        }
                    }
                }
            }
        }
    }

    dev_shear_stress_lj += shear_stress[0];
    dev_shear_stress_rot += shear_stress[1];

    if (SW_PT == rigid) {
        double dmy_shear = 0.0;
#pragma omp parallel for reduction(+ : dmy_shear)
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            double IinvN[DIM] = {0.0, 0.0, 0.0};
            M_v_prod(IinvN, Rigid_IMoments[rigidID][0], torqueGrs[rigidID]);
            const double Jyy =
                (Rigid_Moments[rigidID][2][2] + Rigid_Moments[rigidID][0][0] - Rigid_Moments[rigidID][1][1]) / 2.0;
            const double Jzy = (-Rigid_Moments[rigidID][2][1]);
            dmy_shear += (Jyy * IinvN[2] - Jzy * IinvN[1]);
        }
        rigid_shear_stress[1] = dmy_shear;

        rigid_dev_shear_stress_lj += rigid_shear_stress[0];
        rigid_dev_shear_stress_rot += rigid_shear_stress[1];
    }

    free_1d_int(lscl);
    free_1d_int(head);
}

void Calc_f_Lennard_Jones_shear_cap_primitive(
    Particle *p,
    void (*distance0_func)(const double *x1, const double *x2, double &r12, double *x12),
    const double cap) {
    // Particle 変数の f に
    // !! +=
    //で足す. f の初期値 が正しいと仮定している!!
    const double pair_cutoff = A_R_cutoff * LJ_dia;

    double ss0 = 0.0;
    double rs0 = 0.0;

    for (int n = 0; n < Particle_Number; n++) {
        Particle *p_n       = &p[n];
        int       rigidID_n = -1;
        if (SW_PT == rigid) rigidID_n = Particle_RigidID[n];

        double sum_dmy_fn_0 = 0.0;
        double sum_dmy_fn_1 = 0.0;
        double sum_dmy_fn_2 = 0.0;

        double sum_torqueGrs_0 = 0.0;
        double sum_torqueGrs_1 = 0.0;
        double sum_torqueGrs_2 = 0.0;

#pragma omp parallel for reduction(+: ss0, rs0, sum_dmy_fn_0, sum_dmy_fn_1, sum_dmy_fn_2, sum_torqueGrs_0, sum_torqueGrs_1, sum_torqueGrs_2)
        for (int m = n + 1; m < Particle_Number; m++) {
            Particle *p_m           = &p[m];
            double    r_ij_vec[DIM] = {0.0, 0.0, 0.0};
            double    r_ij          = 0.0;

            distance0_func((*p_n).x, (*p_m).x, r_ij, r_ij_vec);

            if (r_ij < pair_cutoff && !rigid_chain(n, m) && !obstacle_chain((*p_n).spec, (*p_m).spec)) {
                double dmy_r = MIN(cap / r_ij, Lennard_Jones_f(r_ij, LJ_dia, EPSILON, LJ_powers));

                {
                    // forces
                    double dmy_fn[DIM] = {0.0, 0.0, 0.0};
                    for (int d = 0; d < DIM; d++) {
                        dmy_fn[d] = (dmy_r) * (-r_ij_vec[d]);
                        (*p_m).fr[d] -= dmy_fn[d];
                    }
                    sum_dmy_fn_0 += dmy_fn[0];
                    sum_dmy_fn_1 += dmy_fn[1];
                    sum_dmy_fn_2 += dmy_fn[2];

                    // stress
                    ss0 += (dmy_fn[0] * r_ij_vec[1]);

                    // rigid body forces & torques
                    if (SW_PT == rigid) {
                        int rigidID_m = Particle_RigidID[m];

                        for (int d = 0; d < DIM; d++) {
                            forceGrs[rigidID_m][d] -= dmy_fn[d];
                        }

                        sum_torqueGrs_0 += ((GRvecs[n][1] * dmy_fn[2] - GRvecs[n][2] * dmy_fn[1]));
                        sum_torqueGrs_1 += ((GRvecs[n][2] * dmy_fn[0] - GRvecs[n][0] * dmy_fn[2]));
                        sum_torqueGrs_2 += ((GRvecs[n][0] * dmy_fn[1] - GRvecs[n][1] * dmy_fn[0]));

                        torqueGrs[rigidID_m][0] += ((GRvecs[m][1] * dmy_fn[2] - GRvecs[m][2] * dmy_fn[1]));
                        torqueGrs[rigidID_m][1] += ((GRvecs[m][2] * dmy_fn[0] - GRvecs[m][0] * dmy_fn[2]));
                        torqueGrs[rigidID_m][2] += ((GRvecs[m][0] * dmy_fn[1] - GRvecs[m][1] * dmy_fn[0]));

                        double R_IJ_vec[DIM];
                        double R_IJ;
                        distance0_func(xGs[rigidID_n], xGs[rigidID_m], R_IJ, R_IJ_vec);
                        rs0 += (dmy_fn[0] * R_IJ_vec[1]);
                    }
                }
            }
        }

        if (SW_PT == rigid) {
            forceGrs[rigidID_n][0] += sum_dmy_fn_0;
            forceGrs[rigidID_n][1] += sum_dmy_fn_1;
            forceGrs[rigidID_n][2] += sum_dmy_fn_2;

            torqueGrs[rigidID_n][0] += sum_torqueGrs_0;
            torqueGrs[rigidID_n][1] += sum_torqueGrs_1;
            torqueGrs[rigidID_n][2] += sum_torqueGrs_2;
        }
        (*p_n).fr[0] += sum_dmy_fn_0;
        (*p_n).fr[1] += sum_dmy_fn_1;
        (*p_n).fr[2] += sum_dmy_fn_2;
    }

    dev_shear_stress_lj += ss0;
    // dev_shear_stress_rot += shear_stress[1];

    if (SW_PT == rigid) {
        double dmy_shear = 0.0;
#pragma omp parallel for reduction(+ : dmy_shear)
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            double IinvN[DIM] = {0.0, 0.0, 0.0};
            M_v_prod(IinvN, Rigid_IMoments[rigidID][0], torqueGrs[rigidID]);
            const double Jyy =
                (Rigid_Moments[rigidID][2][2] + Rigid_Moments[rigidID][0][0] - Rigid_Moments[rigidID][1][1]) / 2.0;
            const double Jzy = (-Rigid_Moments[rigidID][2][1]);
            dmy_shear += (Jyy * IinvN[2] - Jzy * IinvN[1]);
        }
        rigid_dev_shear_stress_lj += rs0;
        rigid_dev_shear_stress_rot += dmy_shear;
    }
}

void Add_f_gravity(Particle *p) {
    static const double Gravity_on_fluid = G * RHO * 4. / 3. * M_PI * SQ(RADIUS) * RADIUS;
    // Particle 変数の f に
    // !! +=
    //で足す. f の初期値 が正しいと仮定している!!
    if (SW_PT != rigid) {
#pragma omp parallel for
        for (int n = 0; n < Particle_Number; n++) {
            p[n].fr[G_direction] -= Gravity_on_fluid * (MASS_RATIOS[p[n].spec] - 1.0);
        }
    } else {
#pragma omp parallel for
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            const int    rigid_spec   = RigidID_Components[rigidID];
            const double rigid_volume = Rigid_Masses[rigidID] / RHO_particle[rigid_spec];
            forceGrs[rigidID][G_direction] -= G * RHO * rigid_volume * (MASS_RATIOS[rigid_spec] - 1.0);
        }
    }
}
void Calc_f_slip_correct_precision(Particle *p, double const *const *u, const CTime &jikan) {
    static const double dmy0 = DX3 * RHO;
    double              dmy  = dmy0 / jikan.dt_fluid;
    int *               nlattice;
    nlattice = Ns;

    double xp[DIM];
    int    x_int[DIM];
    double residue[DIM];
    int    sw_in_cell;
    double force[DIM];
    double torque[DIM];
    int    r_mesh[DIM];
    double r[DIM];
    double dmy_fp[DIM];
    double x[DIM];
    double dmyR;
    double dmy_phi;
    int    pspec;
#pragma omp parallel for private( \
    xp, x_int, residue, sw_in_cell, force, torque, r_mesh, r, dmy_fp, x, dmyR, dmy_phi, pspec)
    for (int n = 0; n < Particle_Number; n++) {
        for (int d = 0; d < DIM; d++) {
            xp[d] = p[n].x[d];

            force[d] = torque[d] = 0.0;
        }

        sw_in_cell = Particle_cell(xp, DX, x_int, residue);
        sw_in_cell = 1;

        for (int mesh = 0; mesh < NP_domain; mesh++) {
            Relative_coord(Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
            for (int d = 0; d < DIM; d++) {
                x[d] = r_mesh[d] * DX;
            }
            dmyR    = Distance(x, xp);
            dmy_phi = Phi(dmyR, RADIUS);

            int im = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2];
            for (int d = 0; d < DIM; d++) {
                dmy_fp[d] = u[d][im] * dmy_phi;
                force[d] += dmy_fp[d];
            }
            {  // torque
                torque[0] += (r[1] * dmy_fp[2] - r[2] * dmy_fp[1]);
                torque[1] += (r[2] * dmy_fp[0] - r[0] * dmy_fp[2]);
                torque[2] += (r[0] * dmy_fp[1] - r[1] * dmy_fp[0]);
            }
        }  // mesh

        pspec = p[n].spec;
        for (int d = 0; d < DIM; d++) {
            p[n].f_slip[d]      = (dmy * force[d]);
            p[n].torque_slip[d] = (dmy * torque[d]);
        }
    }  // Particle_number
}
void Calc_f_hydro_correct_precision(Particle *p, double const *phi_sum, double const *const *u, const CTime &jikan) {
    static const double dmy0 = -DX3 * RHO;
    double              dmy  = dmy0 / jikan.dt_fluid;

    int *nlattice;
    // if (SW_EQ == Shear_Navier_Stokes){
    // nlattice = Ns_shear;
    //}else {
    nlattice = Ns;
    //}
    double xp[DIM], vp[DIM], omega_p[DIM];
    int    x_int[DIM];
    double residue[DIM];
    int    sw_in_cell;
    double force[DIM];
    double torque[DIM];
    int    r_mesh[DIM];
    double r[DIM];
    double dmy_fp[DIM];
    double x[DIM];
    double dmyR;
    double dmy_phi;
    double v_rot[DIM];
    int    pspec;

    double forceg[DIM];
    double torqueg[DIM];
    int    rigidID;
    // initialize forceGs and torqueGs
    for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
        for (int d = 0; d < DIM; d++) {
            forceGs[rigidID][d]  = 0.0;
            torqueGs[rigidID][d] = 0.0;
        }
    }

#pragma omp parallel for private(xp,         \
                                 vp,         \
                                 omega_p,    \
                                 x_int,      \
                                 residue,    \
                                 sw_in_cell, \
                                 force,      \
                                 torque,     \
                                 r_mesh,     \
                                 r,          \
                                 dmy_fp,     \
                                 x,          \
                                 dmyR,       \
                                 dmy_phi,    \
                                 v_rot,      \
                                 pspec,      \
                                 rigidID,    \
                                 forceg,     \
                                 torqueg)
    for (int n = 0; n < Particle_Number; n++) {
        // double xp[DIM],vp[DIM],omega_p[DIM];
        // int x_int[DIM];
        // double residue[DIM];
        if (SW_PT == rigid) rigidID = Particle_RigidID[n];
        for (int d = 0; d < DIM; d++) {
            xp[d]      = p[n].x[d];
            vp[d]      = p[n].v[d];
            omega_p[d] = p[n].omega[d];

            force[d] = torque[d] = 0.0;
            forceg[d] = torqueg[d] = 0.0;
        }

        sw_in_cell = Particle_cell(xp, DX, x_int, residue);  // {1,0} が返ってくる
        sw_in_cell = 1;
        for (int mesh = 0; mesh < NP_domain; mesh++) {
            Relative_coord(Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
            for (int d = 0; d < DIM; d++) {
                x[d] = r_mesh[d] * DX;
                // dmyR += SQ(r[d]);
            }
            // dmyR = sqrt(dmyR); // vesion2.10 needs this value
            dmyR = Distance(x, xp);  // vesion2.00 needs this value
            Angular2v(omega_p, r, v_rot);

            int im  = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2];
            dmy_phi = Phi(dmyR, RADIUS) / MAX(phi_sum[im], 1.0);
            for (int d = 0; d < DIM; d++) {
                dmy_fp[d] = ((vp[d] + v_rot[d]) - u[d][im]) * dmy_phi;
                force[d] += dmy_fp[d];
            }
            {  // torque
                torque[0] += (r[1] * dmy_fp[2] - r[2] * dmy_fp[1]);
                torque[1] += (r[2] * dmy_fp[0] - r[0] * dmy_fp[2]);
                torque[2] += (r[0] * dmy_fp[1] - r[1] * dmy_fp[0]);
            }

            if (SW_PT == rigid) {
                for (int d = 0; d < DIM; d++) forceg[d] += dmy_fp[d];
                torqueg[0] += ((GRvecs[n][1] + r[1]) * dmy_fp[2] - (GRvecs[n][2] + r[2]) * dmy_fp[1]);
                torqueg[1] += ((GRvecs[n][2] + r[2]) * dmy_fp[0] - (GRvecs[n][0] + r[0]) * dmy_fp[2]);
                torqueg[2] += ((GRvecs[n][0] + r[0]) * dmy_fp[1] - (GRvecs[n][1] + r[1]) * dmy_fp[0]);
            }
        }  // mesh

        pspec = p[n].spec;

        for (int d = 0; d < DIM; d++) {
            p[n].f_hydro[d] = (dmy * force[d]);
            p[n].f_slip[d]  = 0.0;
        }
        if (ROTATION) {
            for (int d = 0; d < DIM; d++) {
                p[n].torque_hydro[d] = (dmy * torque[d]);
                p[n].torque_slip[d]  = 0.0;
            }
        }
        if (SW_PT == rigid) {
            for (int d = 0; d < DIM; d++) {
#pragma omp atomic
                forceGs[rigidID][d] += dmy * forceg[d];

#pragma omp atomic
                torqueGs[rigidID][d] += dmy * torqueg[d];
            }
        }
    }  // Particle_Number
}

void Calc_f_hydro_correct_precision_OBL(Particle *           p,
                                        double const *       phi_sum,
                                        double const *const *u,
                                        const CTime &        jikan) {
    static const double dmy0 = -DX3 * RHO;
    double              dmy  = dmy0 / jikan.dt_fluid;

    int *nlattice;
    nlattice = Ns;

    double xp[DIM], vp[DIM], omega_p[DIM];
    int    x_int[DIM];
    double residue[DIM];
    int    sw_in_cell;
    double force[DIM];
    double torque[DIM];
    int    r_mesh[DIM];
    double r[DIM];
    double dmy_fp[DIM];
    double x[DIM];
    double dmyR;
    double dmy_phi;
    double dmy_rhop;
    double dmy_ry;
    double v_rot[DIM];
    double volume[Particle_Number];
    double Itrace[Particle_Number];
    int    sign;
    int    im;
    double sum_force  = 0.0;
    double sum_volume = 0.0;

    double forceg[DIM];
    double torqueg[DIM];
    double dVg[Rigid_Number][DIM];
    double dWg[Rigid_Number][DIM];
    // initialize forceGs and torqueGs
    for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
        for (int d = 0; d < DIM; d++) {
            forceGs[rigidID][d]  = 0.0;
            torqueGs[rigidID][d] = 0.0;
        }
    }

    Reset_phi(Hydro_force);
    Reset_phi(Hydro_force_new);
#pragma omp parallel for private(xp,         \
                                 vp,         \
                                 omega_p,    \
                                 x_int,      \
                                 residue,    \
                                 sw_in_cell, \
                                 force,      \
                                 torque,     \
                                 r_mesh,     \
                                 r,          \
                                 dmy_fp,     \
                                 x,          \
                                 dmyR,       \
                                 dmy_phi,    \
                                 dmy_rhop,   \
                                 dmy_ry,     \
                                 v_rot,      \
                                 sign,       \
                                 im,         \
                                 forceg,     \
                                 torqueg)
    for (int n = 0; n < Particle_Number; n++) {
        int rigidID;
        dmy_rhop = RHO_particle[p[n].spec];
        if (SW_PT == rigid) rigidID = Particle_RigidID[n];

        for (int d = 0; d < DIM; d++) {
            xp[d]      = p[n].x[d];
            vp[d]      = p[n].v[d];
            omega_p[d] = p[n].omega[d];

            force[d] = torque[d] = 0.0;
            forceg[d] = torqueg[d] = 0.0;
        }

        volume[n]  = 0.0;
        Itrace[n]  = 0.0;
        sw_in_cell = Particle_cell(xp, DX, x_int, residue);
        sw_in_cell = 1;

        for (int mesh = 0; mesh < NP_domain; mesh++) {
            sign = Relative_coord_check_stepover_Y(
                Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
            dmyR = 0.;
            for (int d = 0; d < DIM; d++) {
                x[d] = r_mesh[d] * DX;
                dmyR += SQ(r[d]);
            }
            im = (r_mesh[0] * NY * NZ_ + r_mesh[1] * NZ_ + r_mesh[2]);

            dmyR    = sqrt(dmyR);
            dmy_phi = Phi(dmyR, RADIUS) / MAX(phi_sum[im], 1.0);

            Angular2v(omega_p, r, v_rot);

            for (int d = 0; d < DIM; d++) {
                if (!(d == 0)) {
                    dmy_fp[d] = ((vp[d] + v_rot[d]) - u[d][im]) * dmy_phi;
                } else {
                    dmy_fp[d] = ((vp[d] - sign * Shear_rate_eff * L_particle[1] + v_rot[d]) - u[d][im]) * dmy_phi;
                }
                force[d] += dmy_fp[d];
            }
            {  // torque
                torque[0] += (r[1] * dmy_fp[2] - r[2] * dmy_fp[1]);
                torque[1] += (r[2] * dmy_fp[0] - r[0] * dmy_fp[2]);
                torque[2] += (r[0] * dmy_fp[1] - r[1] * dmy_fp[0]);
            }
            if (SW_PT == rigid) {
                for (int d = 0; d < DIM; d++) forceg[d] += dmy_fp[d];
                torqueg[0] += ((GRvecs[n][1] + r[1]) * dmy_fp[2] - (GRvecs[n][2] + r[2]) * dmy_fp[1]);
                torqueg[1] += ((GRvecs[n][2] + r[2]) * dmy_fp[0] - (GRvecs[n][0] + r[0]) * dmy_fp[2]);
                torqueg[2] += ((GRvecs[n][0] + r[0]) * dmy_fp[1] - (GRvecs[n][1] + r[1]) * dmy_fp[0]);
            }

            dmy_ry = (r_mesh[1] + sign * L_particle[1]);
#pragma omp atomic
            Hydro_force[im] += dmy_fp[0] * dmy_ry * dmy_rhop;  // viscocity
#pragma omp atomic
            Hydro_force_new[im] += dmy_fp[0] * dmy_ry * dmy_rhop;

            volume[n] += dmy_phi;
            Itrace[n] += dmy_phi * SQ(dmyR);
        }  // mesh
        Itrace[n] *= 2.0 / 3.0;

        for (int d = 0; d < DIM; d++) {
            p[n].f_hydro[d] = (dmy * force[d]);
        }
        if (ROTATION) {
            for (int d = 0; d < DIM; d++) {
                p[n].torque_hydro[d] = (dmy * torque[d]);
            }
        }
        if (SW_PT == rigid) {
            for (int d = 0; d < DIM; d++) {
#pragma omp atomic
                forceGs[rigidID][d] += dmy * forceg[d];

#pragma omp atomic
                torqueGs[rigidID][d] += dmy * torqueg[d];
            }
        }

        for (int mesh = 0; mesh < NP_domain; mesh++) {
            sign = Relative_coord_check_stepover_Y(
                Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
            im = (r_mesh[0] * NY * NZ_ + r_mesh[1] * NZ_ + r_mesh[2]);

            dmyR = 0.;
            for (int d = 0; d < DIM; d++) {
                dmyR += SQ(r[d]);
            }
            dmyR    = sqrt(dmyR);
            dmy_phi = Phi(dmyR, RADIUS) / MAX(phi_sum[im], 1.0);

            dmy_ry              = (r_mesh[1] + sign * L_particle[1]);
            double dmy_stress_p = p[n].momentum_depend_fr[0] / volume[n];

#pragma omp atomic
            Hydro_force[im] -= dmy_stress_p * dmy_ry * dmy_phi;  // viscocity

            if (SW_PT != rigid) {
                double dmy_stress_v = -RHO * force[0] / volume[n];
                double dmy_stress_w = -RHO * (torque[1] * r[2] - torque[2] * r[1]) / Itrace[n];
#pragma omp atomic
                Hydro_force_new[im] += (dmy_stress_v + dmy_stress_w) * dmy_ry * dmy_phi;
            }
        }

#pragma omp critical
        {
            sum_force += force[0];
            sum_volume += volume[n];
        }
    }  // Particle_Number
    sum_volume /= RHO;

    if (SW_PT == rigid) {
#pragma omp parallel for
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            for (int d = 0; d < DIM; d++) {
                dVg[rigidID][d] = jikan.dt_md * Rigid_IMasses[rigidID] * forceGs[rigidID][d];
            }

            int    rigid_first = Rigid_Particle_Cumul[rigidID];
            double Ibody[DIM]  = {Rigid_Moments_body[rigidID][0][0],
                                 Rigid_Moments_body[rigidID][1][1],
                                 Rigid_Moments_body[rigidID][2][2]};
            MD_solver_omega_Euler_update(
                dWg[rigidID], omegaGs[rigidID], torqueGs[rigidID], Ibody, p[rigid_first].q, jikan.dt_md);
        }
    }

#pragma omp parallel for private( \
    xp, x_int, residue, sw_in_cell, r_mesh, r, x, dmyR, dmy_phi, dmy_ry, sign, im, dmy_rhop)
    for (int n = 0; n < Particle_Number; n++) {
        int rigidID;
        dmy_rhop = RHO_particle[p[n].spec];
        for (int d = 0; d < DIM; d++) {
            xp[d] = p[n].x[d];
        }
        sw_in_cell = Particle_cell(xp, DX, x_int, residue);  // {1,0} が返ってくる
        sw_in_cell = 1;

        for (int mesh = 0; mesh < NP_domain; mesh++) {
            sign = Relative_coord_check_stepover_Y(
                Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
            im = (r_mesh[0] * NY * NZ_ + r_mesh[1] * NZ_ + r_mesh[2]);

            dmyR = 0;
            for (int d = 0; d < DIM; d++) {
                x[d] = r_mesh[d] * DX;
                dmyR += SQ(r[d]);
            }
            dmyR    = sqrt(dmyR);
            dmy_phi = Phi(dmyR, RADIUS) / MAX(phi_sum[im], 1.0);

            dmy_ry = (r_mesh[1] + sign * L_particle[1]);
#pragma omp atomic
            Hydro_force[im] -= sum_force * dmy_ry * dmy_phi / sum_volume;  // viscocity

            if (SW_PT == rigid) {
                rigidID             = Particle_RigidID[n];
                double dmy_stress_v = dVg[rigidID][0];
                double dmy_stress_w = dWg[rigidID][1] * (GRvecs[n][2] + r[2]) - dWg[rigidID][2] * (GRvecs[n][1] + r[1]);
#pragma omp atomic
                Hydro_force_new[im] += (dmy_stress_v + dmy_stress_w) * dmy_ry * dmy_phi * dmy_rhop;
            }
        }
    }  // Particle_Number
}

void        Calc_harmonic_torque_quincke(Particle *p) {
#pragma omp parallel for
    for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
        double e_omega_space[DIM] = {0.0, 0.0, 0.0};  // basis vector of the body frame written by space frame
        double harmonic_torque[DIM];

        // get e_space_fix from basis vector of body frame
        rigid_body_rotation(e_omega_space, quincke.e_omega, p[Rigid_Particle_Cumul[rigidID]].q, BODY2SPACE);
        double K_n_dot_e = quincke.K * (quincke.n[0] * e_omega_space[0] + quincke.n[1] * e_omega_space[1] +
                                        quincke.n[2] * e_omega_space[2]);

        harmonic_torque[0] = K_n_dot_e * (quincke.n[1] * e_omega_space[2] - quincke.n[2] * e_omega_space[1]);
        harmonic_torque[1] = K_n_dot_e * (quincke.n[2] * e_omega_space[0] - quincke.n[0] * e_omega_space[2]);
        harmonic_torque[2] = K_n_dot_e * (quincke.n[0] * e_omega_space[1] - quincke.n[1] * e_omega_space[0]);

        for (int d = 0; d < DIM; d++) torqueGrs[rigidID][d] += harmonic_torque[d];
    }
}

void Calc_multipole_interaction_force_torque(Particle *p) {
    {  // Mem copy
#pragma omp parallel for
        for (int i = 0; i < Particle_Number; i++) {
            double *ri = ewald_mem.r[i];
            double *xi = p[i].x;
            for (int d = 0; d < DIM; d++) ri[d] = xi[d];
        }
        if (ewald_param.dipole) {
            for (int specID = 0; specID < Component_Number; specID++) {
                const int nump    = Particle_Numbers[specID];
                double *  mu_body = multipole_mu[specID];
                int       offset  = 0;
#pragma omp parallel for
                for (int i0 = 0; i0 < nump; i0++) {
                    int i = offset + i0;
                    compute_particle_dipole(ewald_mem.mu[i], mu_body, p[i].q);
                }
                offset += Particle_Numbers[specID];
            }
        }
        compute_ewald_sum();
    }

    {  // Update particle forces & torques
#pragma omp parallel for
        for (int i = 0; i < Particle_Number; i++) {
            Particle &pi = p[i];
            for (int d = 0; d < DIM; d++) {
                pi.fr[d] += ewald_mem.force[i][d];
                pi.torque_r[d] += ewald_mem.torque[i][d];
            }
        }
    }

    if (SW_PT == rigid) {  // Update rigid forces & torques
#pragma omp parallel for
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            for (int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID + 1]; n++) {
                const double *frc = ewald_mem.force[n];
                const double *tau = ewald_mem.torque[n];
                for (int d = 0; d < DIM; d++) {
                    forceGrs[rigidID][d] += frc[d];
                    torqueGrs[rigidID][d] += tau[d];
                }
                {
                    torqueGrs[rigidID][0] += (GRvecs[n][1] * frc[2] - GRvecs[n][2] * frc[1]);
                    torqueGrs[rigidID][1] += (GRvecs[n][2] * frc[0] - GRvecs[n][0] * frc[2]);
                    torqueGrs[rigidID][2] += (GRvecs[n][0] * frc[1] - GRvecs[n][1] * frc[0]);
                }
            }
        }
    }
}

void Calc_multipole_interaction_force_torque_with_image(Particle *p) {
    {  // Mem copy
#pragma omp parallel for
        for (int i = 0; i < Particle_Number * 2; i++) {
            double *ri = ewald_mem.r[i];
            double *xi;
            if (i < Particle_Number) {
                xi = p[i].x;
                for (int d = 0; d < DIM; d++) {
                    ri[d] = xi[d];
                }
            } else {
                xi = p[i - Particle_Number].x;
                for (int d = 0; d < DIM; d++) {
                    if (d == 2) {
                        ri[d] = xi[d] - 4.1;
                    } else {
                        ri[d] = xi[d];
                    }
                }   
            }
        }
        if (ewald_param.dipole) {
            for (int specID = 0; specID < Component_Number; specID++) {
                const int nump        = Particle_Numbers[specID];
                const int nump_double = nump * 2;
                double *  mu_body     = multipole_mu[specID];
                int       offset      = 0;
#pragma omp parallel for
                for (int i0 = 0; i0 < nump_double; i0++) {
                    int i = offset + i0;
                    if (i0 < nump) {
                        compute_particle_dipole(ewald_mem.mu[i], mu_body, p[i].q);
                    } else {
                        compute_particle_dipole_image(ewald_mem.mu[i], mu_body, p[i - nump].q);
                    }
                }
                offset += Particle_Numbers[specID] * 2;
            }
        }
        compute_ewald_sum();
    }

    {  // Update particle forces & torques
#pragma omp parallel for
        for (int i = 0; i < Particle_Number; i++) {
            Particle &pi = p[i];
            for (int d = 0; d < DIM; d++) {
                pi.fr[d] += ewald_mem.force[i][d];
                pi.torque_r[d] += ewald_mem.torque[i][d];
            }
        }
    }

    if (SW_PT == rigid) {  // Update rigid forces & torques
#pragma omp parallel for
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            for (int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID + 1]; n++) {
                const double *frc = ewald_mem.force[n];
                const double *tau = ewald_mem.torque[n];
                for (int d = 0; d < DIM; d++) {
                    forceGrs[rigidID][d] += frc[d];
                    torqueGrs[rigidID][d] += tau[d];
                }
                {
                    torqueGrs[rigidID][0] += (GRvecs[n][1] * frc[2] - GRvecs[n][2] * frc[1]);
                    torqueGrs[rigidID][1] += (GRvecs[n][2] * frc[0] - GRvecs[n][0] * frc[2]);
                    torqueGrs[rigidID][2] += (GRvecs[n][0] * frc[1] - GRvecs[n][1] * frc[0]);
                } 
            }
        }
    }
}
