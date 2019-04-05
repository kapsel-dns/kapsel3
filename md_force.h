/*!
  \file md_force.h
  \brief Routines to compute MD forces on particles (header file)
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
 */
#ifndef MD_FORCE_H
#define MD_FORCE_H
#ifdef _MPI
#include <mpi.h>
#endif
#include <assert.h> 
#include "variable.h"
#include "input.h"
#include "interaction.h"
#include "make_phi.h"
#include "particle_rotation_solver.h"

void Add_f_gravity(Particle *p);

/*!
  \brief Compute slip constribution to the hydrodynamic force
  \details \f{align*}{
  \left[\int\df{s}\vec{F}_i^{sq}\right] &= 
  \int\vdf{x}\rho\phi_i\left(\vec{u}^{**} - \vec{u}^{*}\right) \\
  \left[\int\df{s}\vec{N}_i^{sq}\right] &= 
  \int\vdf{x}\rho\phi_i\vec{r}_i\times\left(\vec{u}^{**} - \vec{u}^{*}\right)
  \f}
  \param[in,out] p particle data
  \param[in] u change in fluid velocity field due to slip profile at interfaces
  \param[in] jikan time data
 */
void Calc_f_slip_correct_precision(Particle *p, double const* const* u, const CTime &jikan);

/*!
  \brief Compute hydrodynamic force/torque acting on particles
  \details \f{align*}{
  \left[\int\df{s}\vec{F}_i^H\right] &=  -
  \int\vdf{x}\rho\phi_i\left(\vec{u}_p^n - \vec{u}^{*}\right) \\
  \left[\int\df{s}\vec{N}_i^H\right] &= -
  \int\vdf{x}\rho\phi_i\vec{r}_i\times\left(\vec{u}_p^n - \vec{u}^{*}\right)
  \f}
  \param[in,out] p particle data
  \param[in] u current fluid velocity field
  \param[in] jikan time data
 */
void Calc_f_hydro_correct_precision(Particle *p, double const* phi_sum, double const* const* u, const CTime &jikan);

/*!
  \brief Compute hydrodynamic force acting on particles in a system with oblique coordinates (Lees-Edwards PBC)
 */
void Calc_f_hydro_correct_precision_OBL(Particle *p, double const* phi_sum, double const* const* u, const CTime &jikan);

void Calc_f_Lennard_Jones_shear_cap_primitive_lnk(Particle *p
						  ,void (*distance0_func)(const double *x1,const double *x2,double &r12,double *x12)
						  ,const double cap
						  );


/*!
  \brief Compute Lennard-Jones pairwise forces, as well as the particle contribution to the elastic stress
  \details \f[
  \tensor{J} = -\sum_i \vec{x}_i \vec{F}_i = -\sum_{i<j} \vec{x}_{ij} \vec{F}_{ij}
  \f]
  \param[in,out] p particle data (updated with Lennard-Jones forces)
  \param[in] distance0_func distance function
  \param[in] cap cutoff value for the force (divided by particle distance)
 */
void Calc_f_Lennard_Jones_shear_cap_primitive(Particle *p
					      ,void (*distance0_func)(const double *x1,const double *x2,double &r12,double *x12)
					      ,const double cap
					      );
inline void Calc_f_Lennard_Jones(Particle *p){
//  Calc_f_Lennard_Jones_shear_cap_primitive(p,Distance0,DBL_MAX);
    Calc_f_Lennard_Jones_shear_cap_primitive_lnk(p, Distance0, DBL_MAX);
}
inline void Calc_f_Lennard_Jones_OBL(Particle *p){
  Calc_f_Lennard_Jones_shear_cap_primitive(p, Distance0_OBL, DBL_MAX);
}
inline void Calc_anharmonic_force_chain(Particle *p, void (*distance0_func)(const double *x1, const double *x2,double &r12, double *x12)){
    double anharmonic_spring_cst=30.*EPSILON/SQ(SIGMA);
    const double R0=1.5*SIGMA;
    const double iR0=1./R0;
    const double iR02=SQ(iR0);

    int n_first_chain = 0;    
    double shear_stress = 0.0;

#ifdef _MPI
    Particle_Gather (p, p_tmp, SW_OFF);
#endif
    if (procid == root) {
#ifdef _MPI
        Particle_qsort (p_tmp, Particle_Number);
        for (int i = 0; i < Particle_Number; i++) {
            p[i] = p_tmp[i];
        }
#endif

        for(int i = 0; i < Component_Number; i++){
            for(int j = 0; j < Chain_Numbers[i]; j++){
                for(int k = 0; k < Beads_Numbers[i] - 1; k++){
                    int n = n_first_chain + k;
                    int m = n + 1;
                    //fprintf(stdout, "# %d %d %d %d %d\n", i, j, k, n, m);

                    double dmy_r1[DIM];
                    double dm_r1 = 0.0;
                    distance0_func(p[m].x, p[n].x, dm_r1, dmy_r1);

                    double dm1 = 1.0/(1.0 - SQ(dm_r1)*iR02);
                    if(dm1 < 0.0){
                        fprintf(stderr, "### anharmonic error: %d %d %g\n", n, m, dm_r1);
                    }

                    for(int d = 0; d < DIM; d++){
                        double dmy = dm1 * dmy_r1[d];
                        p[n].fr[d] += (-anharmonic_spring_cst)*dmy;
                        p[m].fr[d] += (anharmonic_spring_cst)*dmy;
                    }
                    shear_stress += ((-anharmonic_spring_cst * dm1 * dmy_r1[0]) * (dmy_r1[1]));
                }// beads
                n_first_chain += Beads_Numbers[i];
            }//chains
        }//species
    }

#ifdef _MPI
    Particle_Group_Communication (p, ONE_TO_MANY);
    MPI_Bcast(&shear_stress, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
#endif

    dev_shear_stress_lj += shear_stress;
}

/*!
  \brief Chain debug info for chains in shear flow (x-y plane)
 */
inline void rigid_chain_debug(Particle *p, const double &time, 
                              const int &rigidID=0){
  if(SW_PT == rigid){
    int pid = Rigid_Particle_Cumul[rigidID];
    fprintf(stdout, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
            ,time
            ,-atan2(GRvecs[pid][1], -GRvecs[pid][0])
            ,p[pid].x[0]
            ,p[pid].x[1]
            ,xGs[rigidID][0]
            ,xGs[rigidID][1]
            ,p[pid].v[0]
            ,p[pid].v[1]
            ,velocityGs[rigidID][0]
            ,velocityGs[rigidID][1]
            ,p[pid].omega[2]
            ,omegaGs[rigidID][2]
            ,forceGs[rigidID][0]
            ,forceGs[rigidID][1]
            ,torqueGs[rigidID][2]
            );
  }
}
#endif
