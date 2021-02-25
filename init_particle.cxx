/*!
  \file init_particle.cxx
  \brief Initialize particle properties
  \details Initializes positions, velocities, forces, torques on particles
  \author Y. Nakayama
  \date 2006/07/28
  \version 1.1
 */

#include "init_particle.h"

void Init_Particle(Particle *p) {
    Particle_domain(Phi, NP_domain, Sekibun_cell);

    // particle properties, velocities, forces, etc.
    {
        int offset = 0;
        SRA(GIVEN_SEED, 10);
        for (int j = 0; j < Component_Number; j++) {
            for (int n = 0; n < Particle_Numbers[j]; n++) {
                int i     = offset + n;
                p[i].spec = j;
            }
            offset += Particle_Numbers[j];
        }
    }

    // redefine available box size in case of walls
    double l_particle[DIM];
    double WALL_EXCLUSION = 0.0;
    for (int d = 0; d < DIM; d++) l_particle[d] = L_particle[d];
    if (SW_WALL == FLAT_WALL) {
        WALL_EXCLUSION        = RADIUS + HXI;
        l_particle[wall.axis] = (wall.hi - wall.lo) - 2.0 * WALL_EXCLUSION;  // free length in perpendicular direction
    }

    if (ROTATION) {
        Angular2v = Angular2v_rot_on;
    } else {
        Angular2v = Angular2v_rot_off;
    }
    if (VF > 1.0) {
        int ok_overlap = 0;
        if (SW_PT == spherical_particle) {
            for (int i = 0; i < Component_Number; i++) {
                if (janus_propulsion[i] == obstacle) ok_overlap = 1;
            }
        } else if (SW_PT == rigid) {
            ok_overlap = 1;
        }
        if (ok_overlap) {
            fprintf(stderr, "# WARNING: volume fraction = %g > 1\n", VF);
        } else {
            fprintf(stderr, "volume fraction = %g > 1\n", VF);
            fprintf(stderr, "too many particles\n");
            exit_job(EXIT_FAILURE);
        }
    }
    if (DISTRIBUTION == None) {  // position
        fprintf(stderr, "#init_particle: configuration directly specified in main().: ");
        fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);
    } else if (DISTRIBUTION == uniform_random) {  // position,  method1
        fprintf(stderr, "#init_particle: uniformly distributed: ");
        fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);
        const double overlap_length = SIGMA * 1.05;
        // const double overlap_length = SIGMA * pow(2.,1./6.);
        for (int i = 0; i < Particle_Number; i++) {
            int overlap = 1;
            do {
                for (int d = 0; d < DIM; d++) {
                    p[i].x[d] = RAx(l_particle[d]);
                }
                int j;
                for (j = 0; j < i; j++) {
                    if (Distance(p[i].x, p[j].x) <= overlap_length) {  // if overlap
                        break;
                    }
                }
                if (j >= i) {  // if not overlap
                    overlap = 0;
                }
            } while (overlap);
        }
    } else if (DISTRIBUTION == FCC || DISTRIBUTION == random_walk) {
        if (DISTRIBUTION == FCC) {
            fprintf(stderr, "#init_particle: distributed on FCC latice: ");
            fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);
        }

        {
            double lratio[DIM];
            {
                double min_l = DBL_MAX;
                for (int d = 0; d < DIM; d++) {
                    min_l = MIN(min_l, l_particle[d]);
                }
                for (int d = 0; d < DIM; d++) {
                    lratio[d] = l_particle[d] / min_l;
                }
            }
            int nn[DIM];
            int nz, nxny;
            {
                double dmy         = pow((double)Particle_Number / (4. * lratio[0] * lratio[1] * lratio[2]), 1. / DIM);
                int    nn_base     = (int)ceil(dmy);
                int    nn_base_up  = (int)ceil(dmy);
                int    nn_base_low = (int)(dmy);
                {
                    int nxny_up  = 2 * SQ(nn_base_up) * (int)lratio[0] * (int)lratio[1];
                    int nxny_low = 2 * SQ(nn_base_low) * (int)lratio[0] * (int)lratio[1];
                    int nz_up    = (int)ceil((double)Particle_Number / nxny_up);
                    int nz_low   = (int)ceil((double)Particle_Number / nxny_low);

                    double density_xy_up  = sqrt((double)nxny_up / (l_particle[0] * l_particle[1]));
                    double density_xy_low = sqrt((double)nxny_low / (l_particle[0] * l_particle[1]));
                    double density_z_up   = nz_up / l_particle[2];
                    double density_z_low  = nz_low / l_particle[2];
                    double skewness_up    = fabs(density_xy_up / density_z_up - 1.);
                    double skewness_low   = fabs(density_xy_low / density_z_low - 1.);
                    {
                        if ((nxny_low > 0 && nz_low > 0) && skewness_low <= skewness_up) {
                            nn_base = nn_base_low;
                        } else if (nxny_up > 0 && nz_up > 0) {
                            nn_base = nn_base_up;
                        }
                        for (int d = 0; d < DIM; d++) {
                            nn[d] = (int)(nn_base * ceil(lratio[d]));
                        }
                    }
                }
            }
            int SW_just_packed;
            if (Particle_Number - 4 * nn[0] * nn[1] * nn[2] == 0) {
                SW_just_packed = 1;
            } else {
                SW_just_packed = 0;
                nxny           = 2 * nn[0] * nn[1];
                nz             = (int)ceil((double)Particle_Number / nxny);
            }
            ///////////////////////////////////
            double lattice[DIM];
            double origin[DIM];
            for (int d = 0; d < DIM; d++) {
                lattice[d] = l_particle[d] / nn[d];
                if (!SW_just_packed && d == 2) {
                    lattice[d] = l_particle[d] / (nz * .5);
                }
                origin[d] = lattice[d] * .25;
                // origin[d] = RADIUS;
                if (lattice[d] < SIGMA * sqrt(2.)) {
                    fprintf(stderr,
                            "beyond closed packing in x%d-direction. lattice[%d]=%g < %g\n",
                            d,
                            d,
                            lattice[d],
                            SIGMA * sqrt(2.));
                    fprintf(stderr, "set the value of A <= %g\n", lattice[d] / sqrt(2.) * .5 / DX);
                    fprintf(stderr, "(closely packed VF = %g) < (VF=%g)\n", M_PI / (3. * sqrt(2.)), VF);
                    exit_job(EXIT_FAILURE);
                }
            }
            for (int i = 0; i < Particle_Number; i++) {
                int zlayer = 2 * nn[0] * nn[1];
                int ix     = i % nn[0];
                int iy     = (i % zlayer) / nn[0];
                int iz     = i / zlayer;

                p[i].x[0] = origin[0] + (double)ix * lattice[0] + (lattice[0] / 2.0) * ((iy + iz) % 2);
                p[i].x[1] = origin[1] + (double)iy * lattice[1] / 2.0;
                p[i].x[2] = origin[2] + (double)iz * lattice[2] / 2.0;
            }
            for (int n = 0; n < Particle_Number; n++) {
                for (int d = 0; d < DIM; d++) {
                    p[n].fr[d]          = 0.0;
                    p[n].fr_previous[d] = 0.0;

                    p[n].torque_r[d]          = 0.0;
                    p[n].torque_r_previous[d] = 0.0;
                }
            }
            const double save_A_R_cutoff = A_R_cutoff;
            {
                if (LJ_powers == 0) {
                    A_R_cutoff = pow(2., 1. / 6.);
                }
                if (LJ_powers == 1) {
                    A_R_cutoff = pow(2., 1. / 12.);
                    fprintf(stderr, "# A_R_cutoff %f\n", A_R_cutoff);
                }
                if (LJ_powers == 2) {
                    A_R_cutoff = pow(2., 1. / 18.);
                    fprintf(stderr, "# A_R_cutoff %f\n", A_R_cutoff);
                }
                if (LJ_powers == 3) {
                    A_R_cutoff = 1.0;
                    fprintf(stderr, "# A_R_cutoff %f\n", A_R_cutoff);
                }
                double zmin = 0.;
                double zmax = l_particle[2];
                double ymin = 0.;
                double ymax = l_particle[1];

                double minz, maxz;
                double miny, maxy;
                int    cnt = 0;
                do {
                    Random_Walk(p, 1.e-3);
                    Steepest_descent(p);
                    minz = DBL_MAX;
                    maxz = 0.;
                    miny = DBL_MAX;
                    maxy = 0.;
                    for (int n = 0; n < Particle_Number; n++) {
                        minz = MIN(p[n].x[2], minz);
                        maxz = MAX(p[n].x[2], maxz);

                        miny = MIN(p[n].x[1], miny);
                        maxy = MAX(p[n].x[1], maxy);
                    }
                    cnt++;
                } while (minz < zmin || maxz > zmax || miny < ymin || maxy > ymax);
                A_R_cutoff = save_A_R_cutoff;
            }
        }
        if (DISTRIBUTION == random_walk) {
            fprintf(stderr, "#init_particle: random walk (%d steps): ", N_iteration_init_distribution);
            fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);

            for (int n = 0; n < Particle_Number; n++) {
                for (int d = 0; d < DIM; d++) {
                    p[n].fr[d]          = 0.0;
                    p[n].fr_previous[d] = 0.0;

                    p[n].torque_r[d]          = 0.0;
                    p[n].torque_r_previous[d] = 0.0;
                }
            }
            {
                const double save_A_R_cutoff = A_R_cutoff;
                {
                    A_R_cutoff = pow(2., 1. / 6.);
                    for (int n = 0; n < N_iteration_init_distribution; n++) {
                        Random_Walk(p);
                        Steepest_descent(p);
                    }
                    A_R_cutoff = save_A_R_cutoff;
                }
            }
        }
    } else if (DISTRIBUTION == user_specify) {
        fprintf(stderr, "############################\n");
        fprintf(stderr, "# init_particle: configuration and velocity specified by user: ");
        fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);

        // double specified_position[Particle_Number][DIM];
        for (int i = 0; i < Particle_Number; i++) {
            char dmy[256];
            sprintf(dmy, "switch.INIT_distribution.user_specify.Particles[%d]", i);
            Location target(dmy);
            sprintf(dmy, "user_specify.Particles[%d]", i);
            ufin->get(target.sub("R.x"), p[i].x[0]);
            ufin->get(target.sub("R.y"), p[i].x[1]);
            ufin->get(target.sub("R.z"), p[i].x[2]);
            ufout->put(target.sub("R.x"), p[i].x[0]);
            ufout->put(target.sub("R.y"), p[i].x[1]);
            ufout->put(target.sub("R.z"), p[i].x[2]);
            ufres->put(target.sub("R.x"), p[i].x[0]);
            ufres->put(target.sub("R.y"), p[i].x[1]);
            ufres->put(target.sub("R.z"), p[i].x[2]);

            ufin->get(target.sub("v.x"), p[i].v[0]);
            ufin->get(target.sub("v.y"), p[i].v[1]);
            ufin->get(target.sub("v.z"), p[i].v[2]);
            ufout->put(target.sub("v.x"), p[i].v[0]);
            ufout->put(target.sub("v.y"), p[i].v[1]);
            ufout->put(target.sub("v.z"), p[i].v[2]);
            ufres->put(target.sub("v.x"), p[i].v[0]);
            ufres->put(target.sub("v.y"), p[i].v[1]);
            ufres->put(target.sub("v.z"), p[i].v[2]);

            double q0, q1, q2, q3;
            ufin->get(target.sub("q.q0"), q0);
            ufin->get(target.sub("q.q1"), q1);
            ufin->get(target.sub("q.q2"), q2);
            ufin->get(target.sub("q.q3"), q3);
            ufout->put(target.sub("q.q0"), q0);
            ufout->put(target.sub("q.q1"), q1);
            ufout->put(target.sub("q.q2"), q2);
            ufout->put(target.sub("q.q3"), q3);
            ufres->put(target.sub("q.q0"), q0);
            ufres->put(target.sub("q.q1"), q1);
            ufres->put(target.sub("q.q2"), q2);
            ufres->put(target.sub("q.q3"), q3);
            qtn_init(p[i].q, q0, q1, q2, q3);

            ufin->get(target.sub("omega.x"), p[i].omega[0]);
            ufin->get(target.sub("omega.y"), p[i].omega[1]);
            ufin->get(target.sub("omega.z"), p[i].omega[2]);
            ufout->put(target.sub("omega.x"), p[i].omega[0]);
            ufout->put(target.sub("omega.y"), p[i].omega[1]);
            ufout->put(target.sub("omega.z"), p[i].omega[2]);
            ufres->put(target.sub("omega.x"), p[i].omega[0]);
            ufres->put(target.sub("omega.y"), p[i].omega[1]);
            ufres->put(target.sub("omega.z"), p[i].omega[2]);

            if (ORIENTATION == user_dir) {
                qtn_normalize(p[i].q);
            } else {
                qtn_init(p[i].q, 1.0, 0.0, 0.0, 0.0);
            }
            qtn_init(p[i].q_old, p[i].q);
        }
        if (!RESUMED) {
            delete ufin;
        }
    } else if (DISTRIBUTION == BCC) {
        fprintf(stderr, "#init_particle: distributed on BCC latice: ");
        fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);
        double dmy = pow((double)Particle_Number / 2., 1. / DIM);
        int    nn  = (int)ceil(dmy);
        double ax  = l_particle[0] / (double)nn;
        double ay  = l_particle[1] / (double)nn;
        double az  = l_particle[2] / (double)nn;

        for (int i = 0; i < Particle_Number; i++) {
            int ix = i % nn;
            int iy = (i % (nn * nn)) / nn;
            int iz = i / (nn * nn);

            p[i].x[0] = ax / 4.0 + (double)ix * ax;
            p[i].x[1] = ay / 4.0 + (double)iy * ay;
            p[i].x[2] = az / 4.0 + (double)iz * az / 2.0;

            int m = iz % 2;
            if (m == 1) {
                p[i].x[0] = p[i].x[0] + ax / 2.0;
                p[i].x[1] = p[i].x[1] + ax / 2.0;
            }
        }
    }

    // set orientation
    if (ROTATION) {
        if (ORIENTATION == random_dir) {
            if (SW_QUINCKE == QUINCKE_OFF) {
                for (int i = 0; i < Particle_Number; i++) random_rqtn(p[i].q);
            } else {
                for (int i = 0; i < Particle_Number; i++) get_quaternion_xy_random_Quincke(p[i].q, quincke.e_dir);
            }
            for (int i = 0; i < Particle_Number; i++) {
                qtn_isnormal(p[i].q);
                qtn_init(p[i].q_old, p[i].q);
            }
        } else if (ORIENTATION == space_dir || (ORIENTATION == user_dir && DISTRIBUTION != user_specify)) {
            for (int i = 0; i < Particle_Number; i++) {
                qtn_init(p[i].q, 1.0, 0.0, 0.0, 0.0);
                qtn_isnormal(p[i].q);
                qtn_init(p[i].q_old, p[i].q);
            }
        } else if (ORIENTATION == user_dir && DISTRIBUTION == user_specify) {
            // do nothing orientation already read

        } else {
            fprintf(stderr, "Error: wrong ORIENTATION\n");
            fprintf(stderr, "%d %d %d\n", ORIENTATION, space_dir, user_dir);
            fprintf(stderr, "%d %d \n", DISTRIBUTION, user_specify);
            exit_job(EXIT_FAILURE);
        }
    } else {
        for (int i = 0; i < Particle_Number; i++) {
            qtn_init(p[i].q, 1.0, 0.0, 0.0, 0.0);
            qtn_isnormal(p[i].q);
            qtn_init(p[i].q_old, p[i].q);
        }
    }

    // correct positions in presence of walls
    if (SW_WALL != NO_WALL) {
        if (DISTRIBUTION == uniform_random || DISTRIBUTION == FCC || DISTRIBUTION == BCC ||
            DISTRIBUTION == random_walk) {
            fprintf(stderr,
                    "# Initial particle positions computed with modified system size : %6.2f %6.2f %6.2f\n",
                    l_particle[0],
                    l_particle[1],
                    l_particle[2]);
            for (int n = 0; n < Particle_Number; n++) p[n].x[wall.axis] += (wall.lo + WALL_EXCLUSION);
        }
        double overlap_distance = RADIUS - HXI;
        if (SW_PT == rigid) {
            for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
                for (int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID + 1]; n++) {
                    double x = p[n].x[wall.axis];
                    if (x < wall.lo + overlap_distance || x > wall.hi - overlap_distance)
                        fprintf(stderr, "# INIT WARNING: rigid particle %d (bead %d) overlaps with wall\n", rigidID, n);
                }
            }
        } else {
            for (int n = 0; n < Particle_Number; n++) {
                double x = p[n].x[wall.axis];
                if (x < wall.lo + overlap_distance || x > wall.hi - overlap_distance)
                    fprintf(stderr, "# INIT WARNING: particle %d overlaps with wall\n", n);
            }
        }
    }

    // particle properties, velocities, forces, etc.
    {
        int offset = 0;
        for (int j = 0; j < Component_Number; j++) {
            for (int n = 0; n < Particle_Numbers[j]; n++) {
                int i             = offset + n;
                p[i].mass         = 0.0;
                p[i].surface_mass = 0.0;
                for (int d = 0; d < DIM; d++) {
                    p[i].x_nopbc[d] = p[i].x[d];

                    if (DISTRIBUTION != user_specify) p[i].v[d] = 0.e0;
                    p[i].v_old[d]             = 0.e0;
                    p[i].v_slip[d]            = 0.0;
                    p[i].f_hydro[d]           = 0.0;
                    p[i].f_hydro_previous[d]  = 0.0;
                    p[i].f_hydro1[d]          = 0.0;
                    p[i].f_slip[d]            = 0.0;
                    p[i].f_slip_previous[d]   = 0.0;
                    p[i].fr[d]                = 0.0;
                    p[i].fr_previous[d]       = 0.0;
                    p[i].torque_r[d]          = 0.0;
                    p[i].torque_r_previous[d] = 0.0;

                    if (DISTRIBUTION != user_specify) p[i].omega[d] = 0.e0;
                    p[i].omega_old[d]             = 0.0;
                    p[i].omega_slip[d]            = 0.0;
                    p[i].torque_hydro[d]          = 0.0;
                    p[i].torque_hydro_previous[d] = 0.0;
                    p[i].torque_hydro1[d]         = 0.0;
                    p[i].torque_slip[d]           = 0.0;
                    p[i].torque_slip_previous[d]  = 0.0;

                    p[i].momentum_depend_fr[d] = 0.0;

                    p[i].mass_center[d]         = 0.0;
                    p[i].surface_mass_center[d] = 0.0;
                    p[i].surface_dv[d]          = 0.0;
                    p[i].surface_dw[d]          = 0.0;
                    for (int l = 0; l < DIM; l++) {
                        p[i].inertia[d][l]         = 0.0;
                        p[i].surface_inertia[d][l] = 0.0;
                    }
                }
            }
            offset += Particle_Numbers[j];
        }
    }

    // initialize rigid status
    if (SW_PT == rigid) {
        // save initial topology : IGNORE PBC and particle overlaps
        init_set_xGs(p);  // compute "trial" com assuming no bead overlap

        // Reset mass moments to account for particle overlap
        Reset_phi(phi);
        Reset_phi(phi_sum);

        if (SW_EQ != Shear_Navier_Stokes_Lees_Edwards && SW_EQ != Shear_Navier_Stokes_Lees_Edwards_FDM &&
            SW_EQ != Shear_NS_LE_CH_FDM) {
            init_set_PBC(p);
            init_set_GRvecs(p);  // compute relative distance vectors with respect to trial com

            Make_phi_particle_sum(phi, phi_sum, p);  // compute overlap factors
            Make_phi_rigid_mass(phi_sum, p);         // compute correct com withouth PBC

            init_set_GRvecs(p);  // compute relative distance vectors without PBC

            Make_phi_rigid_inertia(phi_sum, p);
        } else {
            init_set_PBC_OBL(p);
            init_set_GRvecs(p);  // compute relative distance vectors with respect to trial com

            Make_phi_particle_sum_OBL(phi, phi_sum, p);
            Make_phi_rigid_mass_OBL(phi_sum, p);

            init_set_GRvecs(p);

            Make_phi_rigid_inertia_OBL(phi_sum, p);
        }

        init_Rigid_Coordinates(p);

        init_set_vGs(p);

        double phi_vf = 0.0;
        {  // compute initial volume fraction from phi field
#pragma omp parallel for reduction(+ : phi_vf)
            for (int i = 0; i < NX; i++) {
                for (int j = 0; j < NY; j++) {
                    for (int k = 0; k < NZ; k++) {
                        phi_vf += phi[(i * NY * NZ_) + (j * NZ_) + k];
                    }
                }
            }
        }

        fprintf(stderr, "####\n");
        for (int rigidID = 0; rigidID < Rigid_Number; rigidID++) {
            fprintf(
                stderr, "# Rigid Body %d built from %d spherical beads\n", rigidID, Rigid_Particle_Numbers[rigidID]);
            fprintf(stderr, "# MASS    : %10.6g\n", Rigid_Masses[rigidID]);
            fprintf(stderr, "# COM     : %10.6g %10.6g %10.6g\n", xGs[rigidID][0], xGs[rigidID][1], xGs[rigidID][2]);
            fprintf(stderr, "# MOI     :\n");
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments[rigidID][0][0],
                    Rigid_Moments[rigidID][0][1],
                    Rigid_Moments[rigidID][0][2]);
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments[rigidID][1][0],
                    Rigid_Moments[rigidID][1][1],
                    Rigid_Moments[rigidID][1][2]);
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments[rigidID][2][0],
                    Rigid_Moments[rigidID][2][1],
                    Rigid_Moments[rigidID][2][2]);
            fprintf(stderr, "# MOI_body:\n");
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments_body[rigidID][0][0],
                    Rigid_Moments_body[rigidID][0][1],
                    Rigid_Moments_body[rigidID][0][2]);
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments_body[rigidID][1][0],
                    Rigid_Moments_body[rigidID][1][1],
                    Rigid_Moments_body[rigidID][1][2]);
            fprintf(stderr,
                    "#  %10.4g %10.4g %10.4g\n",
                    Rigid_Moments_body[rigidID][2][0],
                    Rigid_Moments_body[rigidID][2][1],
                    Rigid_Moments_body[rigidID][2][2]);
        }
        fprintf(stderr, "# Volume fraction = Sum phi / V = %10.4g\n", phi_vf * DX3 * RHO * Ivolume);
        fprintf(stderr, "####\n");
    }

    {  // set pinned particle velocities to zero
        if (PINNING && SW_PT != rigid) {
            Pinning(p);
        }
    }
    fprintf(stderr, "############################\n");
}
void Show_parameter(Particle *p) {
    FILE *fp = stderr;
    {
        int dmy = NX * NY * NZ;
        int pow;
        for (pow = 0; dmy > 0; pow++) {
            dmy >>= 1;
        }
        pow -= 1;
        fprintf(fp, "#mesh = %d * %d * %d (= %d >= 2^%d)\n", NX, NY, NZ, NX * NY * NZ, pow);

        fprintf(fp, "#DX = %g:", DX);
        fprintf(fp, " (L_x,L_y,L_z) = %g %g %g\n", L[0], L[1], L[2]);
        fprintf(fp, "#\n");
        if (SW_EQ == Navier_Stokes || SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
            SW_EQ == Navier_Stokes_FDM || SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM ||
            SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
            fprintf(fp, "#(eta, rho, nu) = %g %g %g\n", ETA, RHO, NU);
            fprintf(fp, "# kBT = %g\n", kBT);
            fprintf(fp, "# alpha_v = %g\n", alpha_v);
            fprintf(fp, "# alpha_o = %g\n", alpha_o);
            fprintf(fp, "#\n");
        }
        fprintf(fp, "#(number of particles) = %d\n", Particle_Number);
        fprintf(fp, "#(Radius, xi) = %g %g\n", RADIUS, XI);
    }

    {
        if (SW_EQ == Electrolyte) {
            fprintf(fp, "############################ electrolyte solution\n");
            fprintf(fp, "# (eta, rho, nu) = %g %g %g\n", ETA, RHO, NU);
            fprintf(fp, "# Bjerrum length = %g\n", SQ(Elementary_charge) / (PI4 * kBT * Dielectric_cst));
            fprintf(fp,
                    "# (Dielectric_cst, kBT, Elementary_charge)=(%g, %g, %g)\n",
                    Dielectric_cst,
                    kBT,
                    Elementary_charge);
            fprintf(fp, "#\n");
            double dmy = 0.;
            for (int i = 0; i < Component_Number; i++) {
                fprintf(fp,
                        "# particle species = %d, number of particles = %d, surface charge = %g\n",
                        i,
                        Particle_Numbers[i],
                        Surface_charge[i]);
                dmy -= Surface_charge[i] * Particle_Numbers[i];
            }
            fprintf(fp, "# total charge in solvent = %g\n", dmy * Elementary_charge);
            if (N_spec == 1) {
                fprintf(fp, "# counterion only\n");
                fprintf(fp, "# Valency of counterion = %g\n", Valency_counterion);
                fprintf(fp, "# kinetic coefficient of counterion = %g\n", Onsager_coeff_counterion);
            } else if (N_spec == 2) {
                fprintf(fp, "# Add salt ion\n");
                fprintf(fp, "# Valency of positive ion = %g\n", Valency_positive_ion);
                fprintf(fp, "# Valency of negative ion = %g\n", Valency_negative_ion);
                fprintf(fp, "# kinetic coefficient of positive ion = %g\n", Onsager_coeff_positive_ion);
                fprintf(fp, "# kinetic coefficient of negative ion = %g\n", Onsager_coeff_negative_ion);
                fprintf(fp, "# Debye length = %g\n", Debye_length);
                for (int i = 0; i < Component_Number; i++) {
                    double surface_charge_density =
                        ABS(Surface_charge[i]) * Elementary_charge / (4. * M_PI * SQ(RADIUS));
                    double linear_zeta =
                        RADIUS / (1. + RADIUS / Debye_length) / Dielectric_cst * surface_charge_density;
                    double thermal_potential = kBT / (Valency_positive_ion * Elementary_charge);
                    if (linear_zeta < thermal_potential) {
                        fprintf(fp, "# linear electrostatics regime (for isolated sphere)\n");
                        fprintf(fp, "#  for particle species %d\n", i);
                    } else {
                        fprintf(fp, "# nonlinear electrostatics regime (for isolated sphere)\n");
                        fprintf(fp, "#  for particle species %d\n", i);
                    }
                    fprintf(fp, "#  (linear_potential,kBT/Ze)=(%g,%g)\n", linear_zeta, thermal_potential);
                }
            }
            if (External_field) {
                if (AC) {
                    fprintf(fp,
                            "# AC External electric field Ex= %g, Ey= %g, Ez= %g, Frequency= %g\n",
                            E_ext[0],
                            E_ext[1],
                            E_ext[2],
                            Frequency);
                } else {
                    fprintf(fp, "# DC External electric field Ex= %g, Ey= %g, Ez= %g\n", E_ext[0], E_ext[1], E_ext[2]);
                }
            }
            fprintf(fp, "#\n");
            fprintf(fp, "############################\n");
        }
    }
    {
        fprintf(fp, "# gravitational acceleration= %.10g", G);
        if (G != 0.0) {
            char direction[DIM][32] = {"-X", "-Y", "-Z"};
            fprintf(fp, "\tin %s-direction\n", direction[G_direction]);
        } else {
            fprintf(fp, "\n");
        }
    }
    {
        fprintf(fp, "# total %d steps, sample at every %d steps (%d snapshots)\n", MSTEP, GTS, Num_snap + 1);
        if (SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
            SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
            double total_strain = MSTEP * DT * Shear_rate * LY;
            fprintf(fp, "# total strain = %g (%g Lx)\n", total_strain, total_strain / L[0]);
        }
    }
    {
        fprintf(fp, "#\n");
        fprintf(fp, "# Hydrodynamic interaction -> on\n");
        if (ROTATION) {
            fprintf(fp, "# with rotation of particle\n");
        } else {
            fprintf(fp, "# w/o rotation of particle\n");
        }
        if (FIX_CELL) {
            fprintf(fp, "# time-dependent average pressure gradient ASSIGNED in");
            for (int d = 0; d < DIM; d++) {
                const char *xyz[DIM] = {"x", "y", "z"};
                if (FIX_CELLxyz[d]) {
                    fprintf(fp, " %s-", xyz[d]);
                }
            }
            fprintf(fp, "direction\n");
        }

        if (SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
            SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
            fprintf(fp, "# shear rate = %g\n", Shear_rate);
        }

        if (LJ_truncate >= 0) {
            char line[1 << 10];
            if (LJ_truncate > 0) {
                sprintf(line, "# %s", "repulsive part of LJ");
            } else if (LJ_truncate == 0) {
                sprintf(line, "# attractive LJ (%g sigma)", A_R_cutoff);
            }
            if (LJ_powers == 0) {
                sprintf(line, "%s %s", line, "LJ(12:6)");
            } else if (LJ_powers == 1) {
                sprintf(line, "%s %s", line, "LJ(24:12)");
            } else if (LJ_powers == 2) {
                sprintf(line, "%s %s", line, "LJ(36:18)");
            } else if (LJ_powers == 3) {
                sprintf(line, "%s %s", line, "macro_vdw");
            } else if (LJ_powers == 4) {
                sprintf(line, "%s %s", line, "electro_osmotic_flow");
            } else {
                fprintf(fp, "invalid LJ_powers\n");
                exit_job(EXIT_FAILURE);
            }
            sprintf(line, "%s, EPSILON_LJ= %g", line, EPSILON);
            if (SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
                SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
                if (Srate_depend_LJ_cap < DBL_MAX) {
                    sprintf(line, "%s, cap= %g", line, Srate_depend_LJ_cap);
                }
            }
            fprintf(fp, "%s\n", line);
        } else {
            fprintf(fp, "# no Lennard-Jones force.\n");
        }
        fprintf(fp, "#\n");
        if (SW_EQ == Navier_Stokes) {
            fprintf(fp, "#t_min=1/nu*k_max^2= %g\n", Tdump);
        } else if (SW_EQ == Electrolyte) {
            if (External_field) {
                if (AC) {
                    fprintf(fp, "#t_min=MIN(1/nu*k_max^2, 1/kBT*Onsager_coeff*k_max^2, 1/100*Frequency) %g\n", Tdump);
                }
            } else {
                fprintf(fp, "#t_min=MIN(1/nu*k_max^2, 1/kBT*Onsager_coeff*k_max^2) %g\n", Tdump);
            }
        } else if (SW_EQ == Navier_Stokes_FDM || SW_EQ == Navier_Stokes_Cahn_Hilliard_FDM) {
            fprintf(fp, "#t_min=1/nu*k_max^2= %g\n", Tdump);
        }
        if (SW_EQ != Electrolyte && kBT > 0) {
            fprintf(fp, "#dt_noise= %g\n", DT_noise);
        }
        if (fabs(G) > 0.0) {
            for (int i = 0; i < Component_Number; i++) {
                fprintf(fp,
                        "#interface Stokes time (XI/((2/9)*SQ(RADIUS)/ETA*G* DeltaRHO))= %g\n",
                        XI / ((2. / 9.) * SQ(RADIUS) / ETA * G * (RHO_particle[i] - RHO)));
            }
        }
        if (SW_TIME == AUTO) {
            fprintf(fp, "#dt= %g (acceleration= %g)\n", DT, Axel);
        } else if (SW_TIME == MANUAL) {
            fprintf(fp, "#dt= %g (fixed by user)\n", DT);
        }

        {
            double mass_min = DBL_MAX;
            for (int i = 0; i < Component_Number; i++) {
                mass_min = MIN(mass_min, MASS[i]);
            }
            T_LJ = sqrt(mass_min / EPSILON) * SIGMA;
            fprintf(fp, "#  = %g (LJ time[ (M_{min}/EPSILON)^{0.5} SIGMA])\n", DT / T_LJ);
            if (SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
                SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) {
                fprintf(fp, "#(shear rate * dt) = %g\n", Shear_rate * DT);
                fprintf(fp, "#(LJcap * dt/M_{min}) = %g\n", Srate_depend_LJ_cap * DT / mass_min);
            }
        }

        fprintf(fp, "#sekibun_mesh= %d\n", NP_domain);
        fprintf(fp, "#\n");
    }
    {
        double kmax = MIN(MIN(WAVE_X * TRN_X, WAVE_Y * TRN_Y), WAVE_Z * TRN_Z);
        fprintf(fp, "#k_max * min(RADIUS,xi) = %g (must be >%g)\n", MIN(RADIUS, XI) * kmax, M_PI);
        fprintf(fp, "#\n");
    }

    if ((SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards ||
         SW_EQ == Shear_Navier_Stokes_Lees_Edwards_FDM || SW_EQ == Shear_NS_LE_CH_FDM) &&
        !Fixed_particle) {
        for (int n = 0; n < Particle_Number; n++) {
            p[n].v[0]     = 0.0e0;
            p[n].v_old[0] = 0.0e0;
            if (ROTATION) {
                p[n].omega[2]     = 0.;  // 0.5*Shear_rate;
                p[n].omega_old[2] = 0.;  // 0.5*Shear_rate;
            } else {
                p[n].omega[2]     = 0.;
                p[n].omega_old[2] = 0.;
            }
        }
    }
    /*for (int n = 0; n < Particle_Number; n++){
    for (int d = 0; d < DIM; d++) {
            p[n].x_previous[d] = p[n].x[d];
    }
    }*/
}
void Init_Chain(Particle *p) {
    fprintf(stderr, "#init_particle: Chain distributed randomly ");
    fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);

    const double overlap_length = 0.9 * SIGMA;
    for (int d = 0; d < DIM; d++) {
        p[0].x[d] = HL_particle[d];
    }

    int overlap;

    for (int n = 0; n < Particle_Number - 1; n++) {
        overlap = 1;
        do {
            double dmy0 = 0.96 * SIGMA;
            double dmy1 = RAx(PI2);
            double dmy2 = RAx(M_PI);

            p[n + 1].x[0] = p[n].x[0] + dmy0 * sin(dmy2) * sin(dmy1);
            p[n + 1].x[1] = p[n].x[1] + dmy0 * sin(dmy2) * cos(dmy1);
            p[n + 1].x[2] = p[n].x[2] + dmy0 * cos(dmy2);

            for (int d = 0; d < DIM; d++) {
                p[n + 1].x[d] = fmod(p[n + 1].x[d] + L_particle[d], L_particle[d]);
            }

            int m;
            for (m = 0; m < n + 1; m++) {
                if (Distance(p[m].x, p[n + 1].x) <= overlap_length) {
                    break;
                }
            }
            if (m >= n + 1) {
                overlap = 0;
            }
        } while (overlap);

        qtn_init(p[n].q, 1.0, 0.0, 0.0, 0.0);
        qtn_init(p[n].q_old, p[n].q);
    }
}

void Init_Rigid(Particle *p) {
    fprintf(stderr, "#init_particle: Rigid chain distributed linear ");
    fprintf(stderr, "(VF, VF_LJ) = %g %g\n", VF, VF_LJ);

    double dmy, dmy0, dmy1, dmy2;

    for (int d = 0; d < 1000; d++) dmy = RAx(PI2);

    double overlap_length = 0.9 * SIGMA;

    int rigidID = -1;
    int m, n = 0, rn = 0;
    while (n < Particle_Number) {
        if (rigidID != Particle_RigidID[n] || rn == 0) {
            rigidID += 1;
            while (1) {  // set the 1-st particle of a rigid
                dmy       = RAx(L_particle[0]);
                p[n].x[0] = dmy;
                dmy       = RAx(L_particle[1]);
                p[n].x[1] = dmy;
                dmy       = RAx(L_particle[2]);
                p[n].x[2] = dmy;
                for (int d = 0; d < DIM; d++) p[n].x[d] = fmod(p[n].x[d] + L_particle[d], L_particle[d]);
                fprintf(stderr, "debug0: p[%d]: (%f, %f, %f)\n", n, p[n].x[0], p[n].x[1], p[n].x[2]);
                for (m = 0; m < n; m++) {
                    if (Distance(p[m].x, p[n].x) <= overlap_length) {
                        fprintf(stderr, "debug0: p[%d] and p[%d] overlap...\n", m, n);
                        break;
                    }
                }
                if (m >= n) {
                    rn = 1;
                    break;
                }
            }
        } else if (rn == 1) {
            while (1) {  // set the 2-nd particle of a rigid
                dmy0      = 0.96 * SIGMA;
                dmy1      = RAx(M_PI);
                dmy2      = RAx(PI2);
                p[n].x[0] = p[n - 1].x[0] + dmy0 * sin(dmy1) * cos(dmy2);
                p[n].x[1] = p[n - 1].x[1] + dmy0 * sin(dmy1) * sin(dmy2);
                p[n].x[2] = p[n - 1].x[2] + dmy0 * cos(dmy1);
                for (int d = 0; d < DIM; d++) p[n].x[d] = fmod(p[n].x[d] + L_particle[d], L_particle[d]);
                fprintf(stderr, "debug1: p[%d]: (%f, %f, %f)\n", n, p[n].x[0], p[n].x[1], p[n].x[2]);
                for (m = 0; m < n; m++) {
                    if (Distance(p[m].x, p[n].x) <= overlap_length) {
                        fprintf(stderr, "debug1: p[%d] and p[%d] overlap...\n", m, n);
                        break;
                    }
                }
                if (m >= n) {
                    xGs[rigidID][0] = p[n - 1].x[0] + dmy0 * sin(dmy1) * cos(dmy2);
                    xGs[rigidID][1] = p[n - 1].x[1] + dmy0 * sin(dmy1) * sin(dmy2);
                    xGs[rigidID][2] = p[n - 1].x[2] + dmy0 * cos(dmy1);
                    for (int d = 0; d < DIM; d++) {
                        xGs[rigidID][d]          = fmod(xGs[rigidID][d] + 100. * L_particle[d], L_particle[d]);
                        xGs_nopbc[rigidID][d]    = xGs[rigidID][d];
                        xGs_previous[rigidID][d] = xGs[rigidID][d];
                    }
                    if (Rigid_Particle_Numbers[rigidID] % 2 == 0) {
                        GRvecs[n - 1][0] = -(Rigid_Particle_Numbers[rigidID] / 2 - 0.5) * dmy0 * sin(dmy1) * cos(dmy2);
                        GRvecs[n - 1][1] = -(Rigid_Particle_Numbers[rigidID] / 2 - 0.5) * dmy0 * sin(dmy1) * sin(dmy2);
                        GRvecs[n - 1][2] = -(Rigid_Particle_Numbers[rigidID] / 2 - 0.5) * dmy0 * cos(dmy1);
                    } else {
                        GRvecs[n - 1][0] = -(Rigid_Particle_Numbers[rigidID] / 2) * dmy0 * sin(dmy1) * cos(dmy2);
                        GRvecs[n - 1][1] = -(Rigid_Particle_Numbers[rigidID] / 2) * dmy0 * sin(dmy1) * sin(dmy2);
                        GRvecs[n - 1][2] = -(Rigid_Particle_Numbers[rigidID] / 2) * dmy0 * cos(dmy1);
                    }
                    GRvecs[n][0] = GRvecs[n - 1][0] + dmy0 * sin(dmy1) * cos(dmy2);
                    GRvecs[n][1] = GRvecs[n - 1][1] + dmy0 * sin(dmy1) * sin(dmy2);
                    GRvecs[n][2] = GRvecs[n - 1][2] + dmy0 * cos(dmy1);
                    rn += 1;
                    break;
                }
            }
        } else {
            while (1) {  // set 3-rd...
                p[n].x[0] = p[n - 1].x[0] + dmy0 * sin(dmy1) * cos(dmy2);
                p[n].x[1] = p[n - 1].x[1] + dmy0 * sin(dmy1) * sin(dmy2);
                p[n].x[2] = p[n - 1].x[2] + dmy0 * cos(dmy1);
                for (int d = 0; d < DIM; d++) p[n].x[d] = fmod(p[n].x[d] + L_particle[d], L_particle[d]);
                fprintf(stderr, "debug2: p[%d]: (%f, %f, %f)\n", n, p[n].x[0], p[n].x[1], p[n].x[2]);
                for (m = 0; m < n; m++) {
                    if (Distance(p[m].x, p[n].x) <= overlap_length) {
                        fprintf(stderr, "debug2: p[%d] and p[%d] overlap...\n", m, n);
                        n -= rn + 1;
                        rigidID -= 1;
                        rn = 0;
                        break;
                    }
                }
                if (rn == 0) break;
                if (m >= n) {
                    GRvecs[n][0] = GRvecs[n - 1][0] + dmy0 * sin(dmy1) * cos(dmy2);
                    GRvecs[n][1] = GRvecs[n - 1][1] + dmy0 * sin(dmy1) * sin(dmy2);
                    GRvecs[n][2] = GRvecs[n - 1][2] + dmy0 * cos(dmy1);
                    rn += 1;
                    break;
                }
            }
        }
        n += 1;
    }
    set_Rigid_MMs(p);
    init_Rigid_Coordinates(p);
    init_set_vGs(p);
}
