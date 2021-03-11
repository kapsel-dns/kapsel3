#include "fdm_phase_separation.h"

double * cp;
double * psi;
double * psi_o;
double * psicp;
double * psicp_o;
double * phi_obl;
double **stress;
double **stress_o;

void        Calc_cp(double *phi, double *psi, double *cp) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                double lap_psi = calc_laplacian(psi, im);

                double dphi_dx = calc_gradient_o1_to_o1(phi, im, 0);
                double dphi_dy = calc_gradient_o1_to_o1(phi, im, 1);
                double dphi_dz = calc_gradient_o1_to_o1(phi, im, 2);

                double dpsi_dx = calc_gradient_o1_to_o1(psi, im, 0);
                double dpsi_dy = calc_gradient_o1_to_o1(psi, im, 1);
                double dpsi_dz = calc_gradient_o1_to_o1(psi, im, 2);

                double grad_phi_norm = dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz;

                cp[im] = potential_deriv(psi[im]) - (ps.alpha + 2. * ps.z * phi[im]) * lap_psi -
                         2. * ps.z * (dphi_dx * dpsi_dx + dphi_dy * dpsi_dy + dphi_dz * dpsi_dz) +
                         ps.w * A_XI * grad_phi_norm + 2. * ps.d * (psi[im] - ps.neutral) * phi[im];
            }
        }
    }
}
void        Calc_cp_wall(double *phi, double *phi_p, double *phi_wall, double *psi, double *cp) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                double lap_psi      = calc_laplacian(psi, im);
                double dphi_dx      = calc_gradient_o1_to_o1(phi_p, im, 0);
                double dphi_dy      = calc_gradient_o1_to_o1(phi_p, im, 1);
                double dphi_dz      = calc_gradient_o1_to_o1(phi_p, im, 2);
                double dphi_wall_dx = calc_gradient_o1_to_o1(phi_wall, im, 0);
                double dphi_wall_dy = calc_gradient_o1_to_o1(phi_wall, im, 1);
                double dphi_wall_dz = calc_gradient_o1_to_o1(phi_wall, im, 2);
                double dpsi_dx      = calc_gradient_o1_to_o1(psi, im, 0);
                double dpsi_dy      = calc_gradient_o1_to_o1(psi, im, 1);
                double dpsi_dz      = calc_gradient_o1_to_o1(psi, im, 2);

                double grad_phi_norm = dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz;
                double grad_phi_wall_norm =
                    dphi_wall_dx * dphi_wall_dx + dphi_wall_dy * dphi_wall_dy + dphi_wall_dz * dphi_wall_dz;
                cp[im] = potential_deriv(psi[im]) - (ps.alpha + 2. * ps.z * phi[im]) * lap_psi -
                         2.0 * ps.z * (dphi_dx * dpsi_dx + dphi_dy * dpsi_dy + dphi_dz * dpsi_dz) +
                         ps.w * A_XI * grad_phi_norm + ps.w_wall * A_XI * grad_phi_wall_norm +
                         2. * ps.d * (psi[im] - ps.neutral) * phi[im];
            }
        }
    }
}

void        Calc_cp_OBL(double *phi, double *psi, double *cp, const double degree_oblique) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;

                double lap_psi = calc_laplacian_OBL(psi, im, degree_oblique);

                double dphi_dx_co = calc_gradient_o1_to_o1(phi, im, 0);
                double dphi_dy_co = calc_gradient_o1_to_o1(phi, im, 1);
                double dphi_dz_co = calc_gradient_o1_to_o1(phi, im, 2);

                double dphi_dx_contra =
                    ((1. + degree_oblique * degree_oblique) * dphi_dx_co) - (degree_oblique * dphi_dy_co);
                double dphi_dy_contra = -(degree_oblique * dphi_dx_co) + dphi_dy_co;
                double dphi_dz_contra = dphi_dz_co;

                double grad_phi_norm =
                    dphi_dx_co * dphi_dx_contra + dphi_dy_co * dphi_dy_contra + dphi_dz_co * dphi_dz_contra;

                cp[im] = potential_deriv(psi[im]) - (ps.alpha + 2. * ps.z * phi[im]) * lap_psi +
                         ps.w * A_XI * grad_phi_norm + 2. * ps.d * (psi[im] - ps.neutral) * phi[im];
            }
        }
    }
}

void        Cp2stress(double *cp, double *psi, double **stress) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im        = (i * NY * NZ_) + (j * NZ_) + k;
                stress[0][im] = psi[im] * calc_gradient_o1_to_o1(cp, im, 0) * IRHO;
                stress[1][im] = psi[im] * calc_gradient_o1_to_o1(cp, im, 1) * IRHO;
                stress[2][im] = psi[im] * calc_gradient_o1_to_o1(cp, im, 2) * IRHO;
            }
        }
    }
}

void        Cp2stress_OBL(double *cp, double *psi, double **stress, const double degree_oblique) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im        = (i * NY * NZ_) + (j * NZ_) + k;
                double dcp_dx_co = calc_gradient_o1_to_o1(cp, im, 0);
                double dcp_dy_co = calc_gradient_o1_to_o1(cp, im, 1);
                double dcp_dz_co = calc_gradient_o1_to_o1(cp, im, 2);

                double dmy = psi[im] / RHO;

                stress[0][im] = dmy * ((1. + degree_oblique * degree_oblique) * dcp_dx_co - degree_oblique * dcp_dy_co);
                stress[1][im] = dmy * (-degree_oblique * dcp_dx_co + dcp_dy_co);
                stress[2][im] = dmy * dcp_dz_co;
            }
        }
    }
}

void        Set_poisson_rhs_ps(double **u, double **adv, double **lap, double **stress, double *s, CTime &jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                for (int d = 0; d < DIM; d++) {
                    w_v3_3[d][im] = adv[d][im] - NU * lap[d][im] + stress[d][im];
                }
            }
        }
    }
    Set_poisson_rhs_sub(u, w_v3_3, s, jikan);
}

void        Set_poisson_rhs_viscosity(double **u,
                                      double **u_s,
                                      double **adv,
                                      double **lap,
                                      double **stress,
                                      double * eta_s,
                                      double * s,
                                      CTime &  jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                for (int d = 0; d < DIM; d++) {
                    double nu = eta_s[im] * IRHO;
                    double vt = 0.;
                    switch (d) {
                        case 0:
                            vt = 2. * calc_gradient_o1_to_o1(eta_s, im, 0) * calc_gradient_o1_to_o1(u_s[0], im, 0) +
                                 calc_gradient_o1_to_o1(eta_s, im, 1) *
                                     (calc_gradient_o1_to_o1(u_s[0], im, 1) + calc_gradient_o1_to_o1(u_s[1], im, 0)) +
                                 calc_gradient_o1_to_o1(eta_s, im, 2) *
                                     (calc_gradient_o1_to_o1(u_s[0], im, 2) + calc_gradient_o1_to_o1(u_s[2], im, 0));
                            break;
                        case 1:
                            vt = 2. * calc_gradient_o1_to_o1(eta_s, im, 1) * calc_gradient_o1_to_o1(u_s[1], im, 1) +
                                 calc_gradient_o1_to_o1(eta_s, im, 0) *
                                     (calc_gradient_o1_to_o1(u_s[0], im, 1) + calc_gradient_o1_to_o1(u_s[1], im, 0)) +
                                 calc_gradient_o1_to_o1(eta_s, im, 2) *
                                     (calc_gradient_o1_to_o1(u_s[1], im, 2) + calc_gradient_o1_to_o1(u_s[2], im, 1));
                            break;
                        case 2:
                            vt = 2. * calc_gradient_o1_to_o1(eta_s, im, 2) * calc_gradient_o1_to_o1(u_s[2], im, 2) +
                                 calc_gradient_o1_to_o1(eta_s, im, 0) *
                                     (calc_gradient_o1_to_o1(u_s[0], im, 2) + calc_gradient_o1_to_o1(u_s[2], im, 0)) +
                                 calc_gradient_o1_to_o1(eta_s, im, 1) *
                                     (calc_gradient_o1_to_o1(u_s[1], im, 2) + calc_gradient_o1_to_o1(u_s[2], im, 1));
                    }
                    vt *= IRHO;
                    w_v3_3[d][im] = adv[d][im] - nu * lap[d][im] - vt + stress[d][im];
                }
            }
        }
    }
    Set_poisson_rhs_sub(u, w_v3_3, s, jikan);
}

void Set_poisson_rhs_viscosity_OBL(double **u,
                                   double **u_s,
                                   double **adv,
                                   double **lap,
                                   double **stress,
                                   double * eta_s,
                                   double * s,
                                   CTime &  jikan) {
    const double gt = degree_oblique + Shear_rate_eff * jikan.hdt_fluid;

    const double g11 = 1. + gt * gt;
    const double g12 = -gt;
    const double g21 = -gt;
    const double g22 = 1.;
    const double g33 = 1.;

#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im = (i * NY * NZ_) + (j * NZ_) + k;
                double nu = eta_s[im] * IRHO;

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

                for (int d = 0; d < DIM; d++) {
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
                            vt = (g11 * eta_x + g21 * eta_y) * u_zx + (g12 * eta_x + g22 * eta_y) * u_zy +
                                 g33 * eta_x * u_xz + g33 * eta_y * u_yz + 2. * g33 * eta_z * u_zz;
                            break;
                    }
                    vt *= IRHO;
                    w_v3_3[d][im] = adv[d][im] - nu * lap[d][im] - vt + stress[d][im];
                }
            }
        }
    }
    Set_poisson_rhs_sub(u, w_v3_3, s, jikan);
}
void        Update_u_stress_euler(double **u, double **stress, CTime &jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] -= jikan.dt_fluid * stress[0][im];
                u[1][im] -= jikan.dt_fluid * stress[1][im];
                u[2][im] -= jikan.dt_fluid * stress[2][im];
            }
        }
    }
}
void        Update_u_stress_ab2(double **u, double **stress, double **stress_o, CTime &jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im = (i * NY * NZ_) + (j * NZ_) + k;
                u[0][im] -= 0.5 * jikan.dt_fluid * (3. * stress[0][im] - stress_o[0][im]);
                u[1][im] -= 0.5 * jikan.dt_fluid * (3. * stress[1][im] - stress_o[1][im]);
                u[2][im] -= 0.5 * jikan.dt_fluid * (3. * stress[2][im] - stress_o[2][im]);
            }
        }
    }
}
void        Update_psi_euler(double *psi, double **u, double *phi, double *cp, CTime &jikan) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                for (int d = 0; d < DIM; d++) {
                    int im        = (i * NY * NZ_) + (j * NZ_) + k;
                    w_v3_3[d][im] = psi[im] * u[d][im];
                }
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im             = (i * NY * NZ_) + (j * NZ_) + k;
                double lap_term       = ps.kappa * calc_laplacian(cp, im);
                double advective_term = calc_gradient_o1_to_o1(w_v3_3[0], im, 0) +
                                        calc_gradient_o1_to_o1(w_v3_3[1], im, 1) +
                                        calc_gradient_o1_to_o1(w_v3_3[2], im, 2);
                psi[im] += jikan.dt_fluid * (-advective_term + lap_term);
            }
        }
    }
}

void        Update_psi_euler_OBL(double *psi, double **u, double *cp, CTime &jikan, const double degree_oblique) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                for (int d = 0; d < DIM; d++) {
                    int im        = (i * NY * NZ_) + (j * NZ_) + k;
                    w_v3_3[d][im] = psi[im] * u[d][im];
                }
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im             = (i * NY * NZ_) + (j * NZ_) + k;
                double lap_term       = ps.kappa * calc_laplacian_OBL(cp, im, degree_oblique);
                double advective_term = calc_gradient_o1_to_o1(w_v3_3[0], im, 0) +
                                        calc_gradient_o1_to_o1(w_v3_3[1], im, 1) +
                                        calc_gradient_o1_to_o1(w_v3_3[2], im, 2);
                psi[im] += jikan.dt_fluid * (-advective_term + lap_term);
            }
        }
    }
}

void Init_phase_separation(double *phi, double *psi) {
    std::cout << "#################################" << std::endl;
    if (SW_POTENTIAL == Landau) {
        std::cout << "# Landau potential is selected." << std::endl;
    } else if (SW_POTENTIAL == Flory_Huggins) {
        std::cout << "# Flory-Huggins potential is selected." << std::endl;
    }
    std::cout << "# Parameters" << std::endl;
    std::cout << "# Composition ratio: " << ps.ratio << std::endl;
    std::cout << "# Initial fluctuation: " << ps.init_fluct << std::endl;
    if (SW_POTENTIAL == Landau) {
        std::cout << "# a: " << gl.a << std::endl;
        std::cout << "# b: " << gl.b << std::endl;
    } else if (SW_POTENTIAL == Flory_Huggins) {
        std::cout << "# na: " << fh.na << std::endl;
        std::cout << "# nb: " << fh.nb << std::endl;
        std::cout << "# chi: " << fh.chi << std::endl;
    }
    std::cout << "# d: " << ps.d << std::endl;
    std::cout << "# w: " << ps.w << std::endl;
    std::cout << "# alpha: " << ps.alpha << std::endl;
    std::cout << "# kappa: " << ps.kappa << std::endl;
    std::cout << "# z    : " << ps.z << std::endl;
    std::cout << "#################################" << std::endl;

    if (SW_POTENTIAL == Landau) {
        ps.neutral = 0.;
    } else if (SW_POTENTIAL == Flory_Huggins) {
        ps.neutral = 0.5;
    }

    const int seed = 12345;

    std::mt19937                     mt(seed);
    std::uniform_real_distribution<> rand(ps.ratio - ps.init_fluct, ps.ratio + ps.init_fluct);

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im      = (i * NY * NZ_) + (j * NZ_) + k;
                double randval = rand(mt);
                // double randval = (double)mt() / mt.max();
                // randval -= 0.5;
                // randval *= ps.init_fluct;
                // randval += ps.ratio;
                if (SW_POTENTIAL == Landau) {
                    psi[im] = randval * (1. - phi[im]);
                } else if (SW_POTENTIAL == Flory_Huggins) {
                    psi[im] = randval * (1. - phi[im]) + phi[im] * ps.neutral;
                }
            }
        }
    }

    /*
    for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                            int im = (i * NY * NZ_) + (j * NZ_) + k;
                            double r = sqrt((i - NX / 2.)*(i - NX / 2.) + (j - NY / 2.)*(j - NY / 2.));
                            psi[im] = (r < NX *0.398946) ? 1. : -1.;
                    }
            }
    }
    */
    /*
    for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                            int im = (i * NY * NZ_) + (j * NZ_) + k;
                            psi[im] = (j > NY/2.) ? 1. : -1.;
                    }
            }
    }
    */
}

void Output_xdmf_sca(std::string filename, std::string hdffilename, std::string dataname, CTime &jikan) {
    std::stringstream ss_filename;
    ss_filename << filename << ".xmf";
    std::ofstream os_out(ss_filename.str().c_str(), std::ios::out);

    // header
    os_out << R"(<?xml version="1.0" ?>)" << std::endl;
    os_out << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
    os_out << R"(<Xdmf Version="2.0">)" << std::endl;
    os_out << "<Domain>" << std::endl;
    os_out << R"(<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">)" << std::endl;
    os_out << "" << std::endl;

    int istart = 0;
    if (RESUMED) istart = jikan.ts + 1;
    for (int i = istart; i <= MSTEP; i++) {
        if (i % GTS == 0) {
            os_out << R"(<Grid Name="Structured grid" GridType="Uniform">)" << std::endl;
            os_out << R"(<Time Value=")" << i * jikan.dt_fluid << R"(" />)" << std::endl;
            os_out << R"(<Topology Name="tp" TopologyType="3DCORECTMesh" NumberOfElements=")" << NZ << " " << NY << " "
                   << NX << R"("/>)" << std::endl;
            os_out << R"(<Geometry Name="geo" GeometryType="ORIGIN_DXDYDZ">)" << std::endl;
            os_out << R"(<DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)"
                   << std::endl;
            os_out << "0 0 0" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)"
                   << std::endl;
            os_out << DX << " " << DX << " " << DX << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << "</Geometry>" << std::endl;
            os_out << R"(<Attribute Name = ")" << filename << R"(" AttributeType = "Scalar" Center = "Node">)"
                   << std::endl;
            os_out << R"(<DataItem Dimensions=")" << NZ << " " << NY << " " << NX
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/" << dataname << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << "</Attribute>" << std::endl;
            os_out << "</Grid>" << std::endl;
            os_out << std::endl;
        }
    }
    os_out << "</Grid>" << std::endl;
    os_out << "</Domain>" << std::endl;
    os_out << "</Xdmf>" << std::endl;
    os_out.close();
}

void Output_xdmf_vec(std::string filename,
                     std::string hdffilename,
                     std::string dataname_0,
                     std::string dataname_1,
                     std::string dataname_2,
                     CTime &     jikan) {
    std::stringstream ss_filename;
    ss_filename << filename << ".xmf";
    std::ofstream os_out(ss_filename.str().c_str(), std::ios::out);

    // header
    os_out << R"(<?xml version="1.0" ?>)" << std::endl;
    os_out << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
    os_out << R"(<Xdmf Version="2.0">)" << std::endl;
    os_out << "<Domain>" << std::endl;
    os_out << R"(<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">)" << std::endl;
    os_out << "" << std::endl;

    int istart = 0;
    if (RESUMED) istart = jikan.ts + 1;
    for (int i = istart; i <= MSTEP; i++) {
        if (i % GTS == 0) {
            os_out << R"(<Grid Name="Structured grid" GridType="Uniform">)" << std::endl;
            os_out << R"(<Time Value=")" << i * jikan.dt_fluid << R"(" />)" << std::endl;
            os_out << R"(<Topology Name="tp" TopologyType="3DCORECTMesh" NumberOfElements=")" << NZ << " " << NY << " "
                   << NX << R"("/>)" << std::endl;
            os_out << R"(<Geometry Name="geo" GeometryType="ORIGIN_DXDYDZ">)" << std::endl;
            os_out << R"(<DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)"
                   << std::endl;
            os_out << "0 0 0" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)"
                   << std::endl;
            os_out << DX << " " << DX << " " << DX << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << "</Geometry>" << std::endl;

            os_out << R"(<Attribute Name = ")" << filename << R"(" AttributeType = "Vector" Center = "Node">)"
                   << std::endl;
            os_out << R"(<DataItem ItemType="Function" Dimensions=")" << NZ << " " << NY << " " << NX
                   << R"( 3" Function = "JOIN($0, $1, $2) ">)" << std::endl;

            os_out << R"(<DataItem Dimensions=")" << NZ << " " << NY << " " << NX
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/" << dataname_0 << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << NZ << " " << NY << " " << NX
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/" << dataname_1 << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << NZ << " " << NY << " " << NX
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/" << dataname_2 << std::endl;
            os_out << "</DataItem>" << std::endl;

            os_out << "</DataItem>" << std::endl;
            os_out << "</Attribute>" << std::endl;

            os_out << "</Grid>" << std::endl;
            os_out << std::endl;
        }
    }
    os_out << "</Grid>" << std::endl;
    os_out << "</Domain>" << std::endl;
    os_out << "</Xdmf>" << std::endl;
    os_out.close();
}

void Output_xdmf_particle(std::string filename, std::string hdffilename, CTime &jikan) {
    std::stringstream ss_filename;
    ss_filename << filename << ".xmf";
    std::ofstream os_out(ss_filename.str().c_str(), std::ios::out);

    // header
    os_out << R"(<?xml version="1.0" ?>)" << std::endl;
    os_out << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
    os_out << R"(<Xdmf Version="2.0">)" << std::endl;
    os_out << "<Domain>" << std::endl;
    os_out << R"(<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">)" << std::endl;
    os_out << "" << std::endl;

    int istart = 0;
    if (RESUMED) istart = jikan.ts + 1;
    for (int i = istart; i <= MSTEP; i++) {
        if (i % GTS == 0) {
            os_out << R"(<Grid Name="Structured grid" GridType="Uniform">)" << std::endl;
            os_out << R"(<Time Value=")" << i * jikan.dt_fluid << R"(" />)" << std::endl;
            // os_out << R"(<Information Name="degree_oblique" Value=")" << degree_oblique << R"(" />)" << std::endl;
            os_out << R"(<Topology Name="tp" TopologyType="Polyvertex" NumberOfElements=")" << Particle_Number
                   << R"("/>)" << std::endl;
            // os_out << R"(<Geometry Name="geo" GeometryType="ORIGIN_DXDYDZ">)" << std::endl;
            // os_out << R"(<DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)" <<
            // std::endl; os_out << "0 0 0" << std::endl; os_out << "</DataItem>" << std::endl; os_out << R"(<DataItem
            // Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)" << std::endl; os_out << DX
            // << " "
            // << DX << " " << DX << std::endl; os_out << "</DataItem>" << std::endl; os_out << "</Geometry>" <<
            // std::endl;
            os_out << R"(<Geometry GeometryType="XYZ">)" << std::endl;

            // os_out << R"(<Attribute Name = ")" << filename << R"(" AttributeType = "Vector" Center = "Node">)" <<
            // std::endl;
            os_out << R"(<DataItem ItemType="Function" Dimensions=")" << Particle_Number
                   << R"( 3" Function = "JOIN($0, $1, $2) ">)" << std::endl;

            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RX" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RY" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RZ" << std::endl;
            os_out << "</DataItem>" << std::endl;

            os_out << "</DataItem>" << std::endl;
            // os_out << "</Attribute>" << std::endl;

            os_out << "</Geometry>" << std::endl;

            os_out << R"(<Attribute Name = "particle_velocity" AttributeType = "Vector">)" << std::endl;
            os_out << R"(<DataItem ItemType="Function" Dimensions=")" << Particle_Number
                   << R"( 3" Function = "JOIN($0, $1, $2) ">)" << std::endl;

            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VX" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VY" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VZ" << std::endl;
            os_out << "</DataItem>" << std::endl;

            os_out << "</DataItem>" << std::endl;
            os_out << "</Attribute>" << std::endl;

            os_out << "</Grid>" << std::endl;
            os_out << std::endl;
        }
    }
    os_out << "</Grid>" << std::endl;
    os_out << "</Domain>" << std::endl;
    os_out << "</Xdmf>" << std::endl;
    os_out.close();
}

void Output_xdmf_particle_single(std::string filename, std::string hdffilename, CTime &jikan) {
    std::stringstream ss_filename;
    ss_filename << filename << ".xmf";
    std::ofstream os_out(ss_filename.str().c_str(), std::ios::out);

    // header
    os_out << R"(<?xml version="1.0" ?>)" << std::endl;
    os_out << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
    os_out << R"(<Xdmf Version="2.0">)" << std::endl;
    os_out << "<Domain>" << std::endl;
    os_out << R"(<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">)" << std::endl;
    os_out << "" << std::endl;

    int istart = 0;
    if (RESUMED) istart = jikan.ts + 1;
    for (int i = istart; i <= MSTEP; i++) {
        if (i % GTS == 0) {
            os_out << R"(<Grid Name="Structured grid" GridType="Uniform">)" << std::endl;
            os_out << R"(<Time Value=")" << i * jikan.dt_fluid << R"(" />)" << std::endl;
            // os_out << R"(<Information Name="degree_oblique" Value=")" << degree_oblique << R"(" />)" << std::endl;
            os_out << R"(<Topology Name="tp" TopologyType="Polyvertex" NumberOfElements=")" << Particle_Number
                   << R"("/>)" << std::endl;
            // os_out << R"(<Geometry Name="geo" GeometryType="ORIGIN_DXDYDZ">)" << std::endl;
            // os_out << R"(<DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)" <<
            // std::endl; os_out << "0 0 0" << std::endl; os_out << "</DataItem>" << std::endl; os_out << R"(<DataItem
            // Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">)" << std::endl; os_out << DX
            // << " "
            // << DX << " " << DX << std::endl; os_out << "</DataItem>" << std::endl; os_out << "</Geometry>" <<
            // std::endl;
            os_out << R"(<Geometry GeometryType="XYZ">)" << std::endl;

            // os_out << R"(<Attribute Name = ")" << filename << R"(" AttributeType = "Vector" Center = "Node">)" <<
            // std::endl;
            os_out << R"(<DataItem ItemType="Function" Dimensions=")" << Particle_Number + 1
                   << R"( 3" Function = "JOIN($0, $1, $2) ">)" << std::endl;

            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RX" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RY" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "RZ" << std::endl;
            os_out << "</DataItem>" << std::endl;

            os_out << "</DataItem>" << std::endl;
            // os_out << "</Attribute>" << std::endl;

            os_out << "</Geometry>" << std::endl;

            os_out << R"(<Attribute Name = "particle_velocity" AttributeType = "Vector">)" << std::endl;
            os_out << R"(<DataItem ItemType="Function" Dimensions=")" << Particle_Number + 1
                   << R"( 3" Function = "JOIN($0, $1, $2) ">)" << std::endl;

            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VX" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VY" << std::endl;
            os_out << "</DataItem>" << std::endl;
            os_out << R"(<DataItem Dimensions=")" << Particle_Number + 1
                   << R"(" NumberType="Float" Precision="8" Format="HDF">)" << std::endl;
            os_out << hdffilename << "_" << i / GTS << ".h5:/"
                   << "VZ" << std::endl;
            os_out << "</DataItem>" << std::endl;

            os_out << "</DataItem>" << std::endl;
            os_out << "</Attribute>" << std::endl;

            os_out << "</Grid>" << std::endl;
            os_out << std::endl;
        }
    }
    os_out << "</Grid>" << std::endl;
    os_out << "</Domain>" << std::endl;
    os_out << "</Xdmf>" << std::endl;
    os_out.close();
}

int Output_hdf5_sca(std::string filename, std::string dataname, double *data_1d, int count) {
    std::stringstream ss_filename;
    ss_filename << filename << "_" << count << ".h5";

    const H5std_string FILE_NAME(ss_filename.str().c_str());
    const H5std_string DATASET_NAME(dataname);
    const int          RANK = 3;

    // need to allocate contiguous memory
    double *data = new double[NX * NY * NZ];

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im                         = ijk2im(i, j, k);
                data[i + NX * j + NX * NY * k] = data_1d[im];
            }
        }
    }
    // try
    //{
    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();
    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t dimsf[3];  // dataset dimensions
    dimsf[0] = NZ;
    dimsf[1] = NY;
    dimsf[2] = NX;
    H5::DataSpace dataspace(RANK, dimsf);
    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);
    /*
     * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties.
     */
    H5::DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
    /*
     * Write the data to the dataset using default memory space, file
     * space, and transfer properties.
     */

    dataset.write(data, H5::PredType::NATIVE_DOUBLE);
    //}  // end of try block
    // catch failure caused by the H5File operations

    /*
    catch (H5::FileIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
            error.printError();
            return -1;
    }
    */
    delete[] data;

    return 0;  // successfully terminated
}

int Output_hdf5_vec(std::string filename,
                    std::string dataname_0,
                    std::string dataname_1,
                    std::string dataname_2,
                    double **   data_3d,
                    int         count) {
    std::stringstream ss_filename;
    ss_filename << filename << "_" << count << ".h5";

    const H5std_string FILE_NAME(ss_filename.str().c_str());

    const H5std_string DATASET_NAME_0(dataname_0);
    const H5std_string DATASET_NAME_1(dataname_1);
    const H5std_string DATASET_NAME_2(dataname_2);
    const int          RANK = 3;

    // need to allocate contiguous memory
    double *data_0 = new double[NX * NY * NZ];
    double *data_1 = new double[NX * NY * NZ];
    double *data_2 = new double[NX * NY * NZ];

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int im                           = ijk2im(i, j, k);
                data_0[i + NX * j + NX * NY * k] = data_3d[0][im];
                data_1[i + NX * j + NX * NY * k] = data_3d[1][im];
                data_2[i + NX * j + NX * NY * k] = data_3d[2][im];
            }
        }
    }
    // try
    //{
    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();
    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t dimsf[3];  // dataset dimensions
    dimsf[0] = NZ;
    dimsf[1] = NY;
    dimsf[2] = NX;
    H5::DataSpace dataspace(RANK, dimsf);
    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset_0 = file.createDataSet(DATASET_NAME_0, datatype, dataspace);
    dataset_0.write(data_0, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_1 = file.createDataSet(DATASET_NAME_1, datatype, dataspace);
    dataset_1.write(data_1, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_2 = file.createDataSet(DATASET_NAME_2, datatype, dataspace);
    dataset_2.write(data_2, H5::PredType::NATIVE_DOUBLE);

    //}  // end of try block
    // catch failure caused by the H5File operations
    /*
    catch (H5::FileIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
            error.printError();
            return -1;
    }
    */
    delete[] data_0;
    delete[] data_1;
    delete[] data_2;

    return 0;  // successfully terminated
}

int Output_hdf5_particle(std::string filename, Particle *p, int count) {
    std::stringstream ss_filename;
    ss_filename << filename << "_" << count << ".h5";

    const H5std_string FILE_NAME(ss_filename.str().c_str());

    const H5std_string DATASET_NAME_RX("RX");
    const H5std_string DATASET_NAME_RY("RY");
    const H5std_string DATASET_NAME_RZ("RZ");

    const H5std_string DATASET_NAME_VX("VX");
    const H5std_string DATASET_NAME_VY("VY");
    const H5std_string DATASET_NAME_VZ("VZ");

    const H5std_string DATASET_NAME_DO("DEGREE_OBLIQUE");

    const int RANK = 1;

    // need to allocate contiguous memory

    double *RX = new double[Particle_Number];
    double *RY = new double[Particle_Number];
    double *RZ = new double[Particle_Number];

    double *VX = new double[Particle_Number];
    double *VY = new double[Particle_Number];
    double *VZ = new double[Particle_Number];

    for (int i = 0; i < Particle_Number; i++) {
        const Particle &pi = p[i];

        RX[i] = pi.x[0];
        RY[i] = pi.x[1];
        RZ[i] = pi.x[2];

        VX[i] = pi.v[0];
        VY[i] = pi.v[1];
        VZ[i] = pi.v[2];
    }
    // try
    //{
    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();
    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t dimsf[0];  // dataset dimensions
    dimsf[0] = Particle_Number;

    H5::DataSpace dataspace(RANK, dimsf);
    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset_RX = file.createDataSet(DATASET_NAME_RX, datatype, dataspace);
    dataset_RX.write(RX, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_RY = file.createDataSet(DATASET_NAME_RY, datatype, dataspace);
    dataset_RY.write(RY, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_RZ = file.createDataSet(DATASET_NAME_RZ, datatype, dataspace);
    dataset_RZ.write(RZ, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VX = file.createDataSet(DATASET_NAME_VX, datatype, dataspace);
    dataset_VX.write(VX, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VY = file.createDataSet(DATASET_NAME_VY, datatype, dataspace);
    dataset_VY.write(VY, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VZ = file.createDataSet(DATASET_NAME_VZ, datatype, dataspace);
    dataset_VZ.write(VZ, H5::PredType::NATIVE_DOUBLE);

    hsize_t dimsf_DO;
    dimsf_DO = 1;

    H5::DataSpace dataspace_DO(RANK, &dimsf_DO);

    H5::DataSet dataset_DO = file.createDataSet(DATASET_NAME_DO, datatype, dataspace_DO);
    dataset_DO.write(&degree_oblique, H5::PredType::NATIVE_DOUBLE);

    //}  // end of try block
    // catch failure caused by the H5File operations
    /*
    catch (H5::FileIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
            error.printError();
            return -1;
    }
    */
    delete[] RX;
    delete[] RY;
    delete[] RZ;

    delete[] VX;
    delete[] VY;
    delete[] VZ;

    return 0;  // successfully terminated
}

int Output_hdf5_particle_single(std::string filename, Particle *p, int count) {
    std::stringstream ss_filename;
    ss_filename << filename << "_" << count << ".h5";

    const H5std_string FILE_NAME(ss_filename.str().c_str());

    const H5std_string DATASET_NAME_RX("RX");
    const H5std_string DATASET_NAME_RY("RY");
    const H5std_string DATASET_NAME_RZ("RZ");

    const H5std_string DATASET_NAME_VX("VX");
    const H5std_string DATASET_NAME_VY("VY");
    const H5std_string DATASET_NAME_VZ("VZ");

    const H5std_string DATASET_NAME_DO("DEGREE_OBLIQUE");

    const int RANK = 1;

    // need to allocate contiguous memory

    double *RX = new double[Particle_Number + 1];
    double *RY = new double[Particle_Number + 1];
    double *RZ = new double[Particle_Number + 1];

    double *VX = new double[Particle_Number + 1];
    double *VY = new double[Particle_Number + 1];
    double *VZ = new double[Particle_Number + 1];

    for (int i = 0; i < Particle_Number; i++) {
        const Particle &pi = p[i];

        RX[i] = pi.x[0];
        RY[i] = pi.x[1];
        RZ[i] = pi.x[2];

        VX[i] = pi.v[0];
        VY[i] = pi.v[1];
        VZ[i] = pi.v[2];
    }

    RX[1] = -1.0;
    RY[1] = -1.0;
    RZ[1] = -1.0;

    VX[1] = -1.0;
    VY[1] = -1.0;
    VZ[1] = -1.0;

    // try
    //{
    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();
    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t dimsf[0];  // dataset dimensions
    dimsf[0] = Particle_Number + 1;

    H5::DataSpace dataspace(RANK, dimsf);
    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset_RX = file.createDataSet(DATASET_NAME_RX, datatype, dataspace);
    dataset_RX.write(RX, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_RY = file.createDataSet(DATASET_NAME_RY, datatype, dataspace);
    dataset_RY.write(RY, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_RZ = file.createDataSet(DATASET_NAME_RZ, datatype, dataspace);
    dataset_RZ.write(RZ, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VX = file.createDataSet(DATASET_NAME_VX, datatype, dataspace);
    dataset_VX.write(VX, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VY = file.createDataSet(DATASET_NAME_VY, datatype, dataspace);
    dataset_VY.write(VY, H5::PredType::NATIVE_DOUBLE);

    H5::DataSet dataset_VZ = file.createDataSet(DATASET_NAME_VZ, datatype, dataspace);
    dataset_VZ.write(VZ, H5::PredType::NATIVE_DOUBLE);

    hsize_t dimsf_DO;
    dimsf_DO = 1;

    H5::DataSpace dataspace_DO(RANK, &dimsf_DO);

    H5::DataSet dataset_DO = file.createDataSet(DATASET_NAME_DO, datatype, dataspace_DO);
    dataset_DO.write(&degree_oblique, H5::PredType::NATIVE_DOUBLE);

    //}  // end of try block
    /*
       // catch failure caused by the H5File operations
    catch (H5::FileIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
            error.printError();
            return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
            error.printError();
            return -1;
    }
    */
    delete[] RX;
    delete[] RY;
    delete[] RZ;

    delete[] VX;
    delete[] VY;
    delete[] VZ;

    return 0;  // successfully terminated
}

void        Psi2eta(double *psi, double *eta) {
#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int    im = (i * NY * NZ_) + (j * NZ_) + k;
                double dmy;
                if (SW_POTENTIAL == Landau) {
                    dmy = (1. + psi[im]) / 2.;
                } else if (SW_POTENTIAL == Flory_Huggins) {
                    dmy = psi[im];
                }
                if (dmy > 0. && dmy < 1.) {
                    eta[im] = (ETA_A - ETA_B) * dmy + ETA_B;
                } else if (dmy <= 0.) {
                    eta[im] = ETA_B;
                } else {
                    eta[im] = ETA_A;
                }
            }
        }
    }
}

void xdmf_output(CTime jikan) {
    if (PHASE_SEPARATION) {
        Output_xdmf_sca("fluid_phase", "orderparam", "PSI", jikan);
    }
    if (Particle_Number > 0) {
        Output_xdmf_sca("particle_phase", "particle", "PHI", jikan);
        if (Particle_Number == 1) {
            Output_xdmf_particle_single("particle_coordinate", "particle_data", jikan);
        } else {
            Output_xdmf_particle("particle_coordinate", "particle_data", jikan);
        }
    }
    if (SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM || SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM ||
        SW_EQ == Shear_NS_LE_CH_FDM) {
        // Output_xdmf_sca("shear_rate_field", "shear_rate", "GAMMA", jikan);
    }

    Output_xdmf_vec("velocity_field", "velocity", "UX", "UY", "UZ", jikan);
}