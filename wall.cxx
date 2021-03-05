#include "wall.h"

#include "fdm.h"

void Init_Wall(double* phi_wall) {
    if (SW_WALL == FLAT_WALL) {  // print wall params
        char              dmy[128];
        static const char axis[] = {'x', 'y', 'z'};
        sprintf(dmy,
                "phi_flatwall_%dx%dx%d_h%c%d.dat",
                Ns[0],
                Ns[1],
                Ns[2],
                axis[wall.axis],
                static_cast<int>((wall.hi - wall.lo) / DX));

        FILE*  fwall = filecheckopen(dmy, "w");
        double dr    = DX;
        double l     = L[wall.axis];
        double hl    = l / 2.0;
        double r     = 0.0;
        double phiw  = 0.0;
        while (r <= l) {
            phiw = (r < hl ? Phi(r, wall.lo) : 1.0 - Phi(r, wall.hi));
            fprintf(fwall, "%.5f %.5f\n", r, phiw);
            r += dr;
        }
        fclose(fwall);
    }

    {  // compute wall phi field
        double  rijk[DIM] = {0.0, 0.0, 0.0};
        double& r         = rijk[wall.axis];
        double  hl        = HL[wall.axis];
        for (int i = 0; i < NX; i++) {
            rijk[0] = static_cast<double>(i) * DX;

            for (int j = 0; j < NY; j++) {
                rijk[1] = static_cast<double>(j) * DX;

                for (int k = 0; k < NZ; k++) {
                    rijk[2]      = static_cast<double>(k) * DX;
                    int im       = (i * NY * NZ_) + (j * NZ_) + k;
                    phi_wall[im] = (r < hl ? Phi(r, wall.lo) : 1.0 - Phi(r, wall.hi));
                    if (k < hl) {
                        grad_phi_wall[im] = -DPhi_compact_sin(r, wall.lo);
                    } else {
                        grad_phi_wall[im] = DPhi_compact_sin(r, wall.hi);
                    }
                }
            }
        }
    }
}

void Init_bottom_Wall(double* phi_wall_prime, double* grad_phi_wall_prime) {
    if (SW_WALL == FLAT_WALL) {  // print wall params
        char              dmy[128];
        static const char axis[]    = {'x', 'y', 'z'};
        double            b_wall_lo = 0.5 * wall.lo;
        double            b_wall_hi = wall.hi + b_wall_lo;
        sprintf(dmy,
                "phi_bottom_wall_%dx%dx%d_h%c%d.dat",
                Ns[0],
                Ns[1],
                Ns[2],
                axis[wall.axis],
                static_cast<int>((b_wall_hi - b_wall_lo) / DX));

        FILE*  fwall = filecheckopen(dmy, "w");
        double dr    = DX;
        double l     = L[wall.axis];
        double hl    = l / 2.0;
        double r     = 0.0;
        double phiw  = 0.0;
        double rx    = 0;
        while (r <= l) {
            rx   = r - b_wall_lo;
            phiw = (r < hl ? Phi(abs(rx), b_wall_lo) : 1.0 - Phi(abs(rx), b_wall_hi));
            fprintf(fwall, "%.5f %.5f\n", r, phiw);
            r += dr;
        }
        fclose(fwall);
    }

    {  // compute wall phi field
        double  rijk[DIM] = {0.0, 0.0, 0.0};
        double& r         = rijk[wall.axis];
        double  hl        = HL[wall.axis];
        double  b_wall_lo = 0.5 * wall.lo;
        double  b_wall_hi = wall.hi + b_wall_lo;
        int     wall_lo_z = 0;
        int     wall_hi_z = (int)wall.lo;

        for (int i = 0; i < NX; i++) {
            rijk[0] = static_cast<double>(i) * DX;

            for (int j = 0; j < NY; j++) {
                rijk[1] = static_cast<double>(j) * DX;

                for (int k = 0; k < NZ; k++) {
                    rijk[2]            = static_cast<double>(k) * DX;
                    int    im          = (i * NY * NZ_) + (j * NZ_) + k;
                    double rx          = r - b_wall_lo;
                    phi_wall_prime[im] = (r < hl ? Phi(abs(rx), b_wall_lo) : 1.0 - Phi(abs(rx), b_wall_hi));

                    if (k == wall_lo_z) {
                        grad_phi_wall_prime[im] = 1.;
                    } else if (k == wall_hi_z) {
                        grad_phi_wall_prime[im] = -1.;
                    } else {
                        grad_phi_wall_prime[im] = 0.;
                    }
                }
            }
        }
    }
}

void Init_top_Wall(double* phi_wall_double_prime) {
    if (SW_WALL == FLAT_WALL) {  // print wall params
        char              dmy[128];
        static const char axis[]    = {'x', 'y', 'z'};
        double            b_wall_lo = 0.5 * wall.lo;
        double            b_wall_hi = wall.hi + b_wall_lo;
        sprintf(dmy,
                "phi_top_wall_%dx%dx%d_h%c%d.dat",
                Ns[0],
                Ns[1],
                Ns[2],
                axis[wall.axis],
                static_cast<int>((b_wall_hi - b_wall_lo) / DX));

        FILE*  fwall = filecheckopen(dmy, "w");
        double dr    = DX;
        double l     = L[wall.axis];
        double hl    = l / 2.0;
        double r     = 0.0;
        double phiw  = 0.0;
        double rx    = 0;
        while (r <= l) {
            rx   = r + b_wall_lo;
            phiw = (r < hl ? Phi(abs(rx), b_wall_lo) : 1.0 - Phi(abs(rx), b_wall_hi));
            fprintf(fwall, "%.5f %.5f\n", r, phiw);
            r += dr;
        }
        fclose(fwall);
    }

    {  // compute wall phi field
        double  rijk[DIM] = {0.0, 0.0, 0.0};
        double& r         = rijk[wall.axis];
        double  hl        = HL[wall.axis];
        double  b_wall_lo = 0.5 * wall.lo;
        double  b_wall_hi = wall.hi + b_wall_lo;

        for (int i = 0; i < NX; i++) {
            rijk[0] = static_cast<double>(i) * DX;

            for (int j = 0; j < NY; j++) {
                rijk[1] = static_cast<double>(j) * DX;

                for (int k = 0; k < NZ; k++) {
                    rijk[2]                   = static_cast<double>(k) * DX;
                    int    im                 = (i * NY * NZ_) + (j * NZ_) + k;
                    double rx                 = r + b_wall_lo;
                    phi_wall_double_prime[im] = (r < hl ? Phi(abs(rx), b_wall_lo) : 1.0 - Phi(abs(rx), b_wall_hi));
                }
            }
        }
    }
}

/*!
    \brief Compute force coming from flat walls on one particle
*/
inline double Compute_f_wall_single(const double& x, const double& cutoff, const double& offset) {
    double fx = 0.0;
    double h  = x - wall.lo + offset;  // distance to lower mirror particle
    if (h <= cutoff) fx += MIN(DBL_MAX / h, Lennard_Jones_f(h, LJ_dia, wall.EPSILON, wall.LJ_powers)) * h;

    h = wall.hi - x + offset;
    if (h <= cutoff) fx -= MIN(DBL_MAX / h, Lennard_Jones_f(h, LJ_dia, wall.EPSILON, wall.LJ_powers)) * h;
    return fx;
}

/*!
    \brief  Add forces coming from the flat walls to all particles
*/
void Add_f_wall(Particle* p) {
    double cutoff = wall.A_R_cutoff * LJ_dia;
    double offset = 0.5 * LJ_dia;
    if (SW_WALL == FLAT_WALL) {
        if (SW_PT == rigid) {
#pragma omp parallel for
            for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
                double f_h = 0.0;
                for (int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID + 1]; n++) {
                    double fi = Compute_f_wall_single(p[n].x[wall.axis], cutoff, offset);
                    f_h += fi;
                    p[n].fr[wall.axis] += fi;
                }
                double Fh[DIM] = {0.0, 0.0, 0.0};
                Fh[wall.axis]  = f_h;

                forceGrs[rigidID][wall.axis] += f_h;
                torqueGrs[rigidID][0] = GRvecs[rigidID][1] * Fh[2] - GRvecs[rigidID][2] * Fh[1];
                torqueGrs[rigidID][1] = GRvecs[rigidID][2] * Fh[0] - GRvecs[rigidID][0] * Fh[2];
                torqueGrs[rigidID][2] = GRvecs[rigidID][0] * Fh[1] - GRvecs[rigidID][1] * Fh[0];
            }
        } else {
#pragma omp parallel for
            for (int n = 0; n < Particle_Number; n++)
                p[n].fr[wall.axis] += Compute_f_wall_single(p[n].x[wall.axis], cutoff, offset);
        }
    }
}
