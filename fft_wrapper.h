/*!
  \file fft_wrapper.h
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief FFT wrapper routines for reciprocal space calculations (header file)
  \note For simplicity, the equations given here for the various operations in Fourier space refer to the continuous transformation.
 */
#ifndef FFT_WRAPPER_H
#define FFT_WRAPPER_H

#include <assert.h> 
#include <math.h>

#ifdef _MPI
#include <mpi.h>
#endif
#include "periodic_spline.h"
#include "variable.h"
#include "input.h"
#include "aux_field.h"
#include "alloc.h"
#include "macro.h"
#include "memory_model.h"
#include "fft_wrapper_base.h"
#ifdef _OPENMP
#include <omp.h>
#include <mkl_dfti.h>
#include <complex>
#endif

void Free_fft(void);

////////////////////////

/*!
  \brief Initialize FFT routines
  \details Used to initialize any required workspace memory for FFT routines. For the moment only Ooura's FFT can be used 
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

inline void orth2obl(const int& j, const int& i, int& i_oblique, int& i_oblique_plus, double& alpha, double& beta){
    int delta_j = (j + PREV_NPs[REAL][1]) - NY/2;
    double sign = (double)delta_j;
    if (!(delta_j == 0)) sign = sign/fabs(sign);

    i_oblique = (int)(sign*degree_oblique*delta_j)*sign;
    alpha = (degree_oblique*delta_j - i_oblique)*sign;
    beta  = 1.0 - alpha;

    i_oblique      = (int) fmod(i + i_oblique + 4.0*NX, NX);
    i_oblique_plus = (int) fmod(i_oblique + sign + 4.0*NX, NX);
}

inline void obl2orth(const int &j, const int& i, int& i_plus, int& i_oblique, double& alpha, double& beta){
    int delta_j = (j + PREV_NPs[REAL][1]) - NY/2;
    double sign = (double)delta_j;
    if (!(delta_j == 0)) sign = sign/fabs(sign);

    i_oblique = (int)(sign*degree_oblique*delta_j)*sign + sign;
    alpha = (i_oblique - degree_oblique*delta_j)*sign;
    beta  = 1.0 - alpha;

    i_oblique  = (int) fmod(i + i_oblique + 4.0*NX, NX);
    i_plus     = (int) fmod(i + sign + 2.0*NX, NX);
}


/*!
  \brief Transform scalar field from rectangular to oblique coordinates
  \param[in,out] phi scalar (density) field to transform
 */

inline void phi2phi_oblique(double *phi){
    int im;
    int im_ob;
    int im_ob_p;

    Copy_v1(work_v1, phi);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, NZ_, phi, work_v1) private(im, im_ob, im_ob_p)
#endif
    for (int i = 0; i < NPs[REAL][0]; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            int i_oblique, i_oblique_plus;
            double alpha, beta;
            orth2obl(j, i, i_oblique, i_oblique_plus, alpha, beta);
      
            for (int k = 0; k < NPs[REAL][2]; k++) {
                im = REALMODE_ARRAYINDEX(i, j, k);
                im_ob   = REALMODE_ARRAYINDEX((i_oblique), j, k);
                im_ob_p = REALMODE_ARRAYINDEX((i_oblique_plus), j, k);

                phi[im] = (beta*work_v1[im_ob]+alpha*work_v1[im_ob_p]);
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

#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, NZ_, phi, work_v1) private(im, im_ob, im_p)
#endif
    for (int i = 0; i < NPs[REAL][0]; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            int i_plus, i_oblique;
            double alpha, beta;
            obl2orth(j, i, i_plus, i_oblique, alpha, beta);
            for (int k = 0; k < NPs[REAL][2]; k++) {
                im = REALMODE_ARRAYINDEX(i, j, k);
                im_ob   = REALMODE_ARRAYINDEX((i_oblique), j, k);
                im_p    = REALMODE_ARRAYINDEX((i_plus), j, k);

                phi[im_ob] = beta*work_v1[im] + alpha*work_v1[im_p];
            }
        }
    }
}

// Allocate / Deallocate interpolation memory
inline void Init_Transform_obl(){
    if(SW_OBL_INT == spline_int){
        int nthreads = 1;

        uspline = (double***) malloc(sizeof(double**) * nthreads); 
        splineOblique = new splineSystem*[nthreads];

        for(int np = 0; np < nthreads; np++){
            splineInit(splineOblique[np], NPs[REAL][0], DX);
            uspline[np] = (double**) malloc(sizeof(double*) * DIM );

            for(int d = 0; d < DIM; d++) uspline[np][d] = alloc_1d_double(NPs[REAL][0]);
        }
    }
}

inline void Free_Transform_obl(){
    if(SW_OBL_INT == spline_int){
        int nthreads = 1;
        for(int np = 0; np < nthreads; np++){
            for(int d = 0; d < DIM; d++) free_1d_double(uspline[np][d]);
      
            free(uspline[np]);
            splineFree(splineOblique[np]);
        }
        delete[] splineOblique;
        free(uspline);
    }
}

// Periodic spline interpolation
inline void Spline_u_oblique_transform(double **uu, const OBL_TRANSFORM &flag){
    int im, im_ob;
    double dmy_x;
    double delta_y;
    double sign;
    if(flag == oblique2cartesian){ 
        sign = -1.0;
    }else if(flag == cartesian2oblique){
        sign = 1.0;
    }else{
        exit_job(EXIT_FAILURE);
    }

    for(int j = 0; j < NPs[REAL][1]; j++){//original coord
        int np = 0;    

        splineSystem *spl = splineOblique[np];
        double **us0      = uspline[np];

        delta_y = (double)(j - NPs[REAL][1]/2)*DX;

        for (int k = 0; k < NPs[REAL][2]; k++) { //original coord
            //setup interpolation grid
            for (int i = 0; i < NPs[REAL][0]; i++) {//original coord
                im = REALMODE_ARRAYINDEX(i, j, k);
                dmy_x = fmod(i*DX - sign*degree_oblique*delta_y + 4.0*LX, LX); //transformed coord

                //velocity components in transformed basis defined over
                //original grid points x0
                us0[0][i] = uu[0][im] - sign*degree_oblique*uu[1][im];
                us0[1][i] = uu[1][im];
                us0[2][i] = uu[2][im];
            }//i

            //compute interpolated points
            for(int d = DIM - 1; d >= 0; d--){
                splineCompute(spl, us0[d]);

                for(int i = 0; i < NPs[REAL][0]; i++){//transformed coord
                    im = REALMODE_ARRAYINDEX(i, j, k);
                    dmy_x = fmod(i*DX + sign*degree_oblique*delta_y + 4.0*LX, LX); // original coord

                    uu[d][im]  = splineFx(spl, dmy_x);
                    if(d == 0 && sign < 0) uu[d][im] += Shear_rate_eff*delta_y;
                }//i
            }//d
        }//k
    }//j
}

inline void U2u_oblique(double **uu) {
    int im;
    int im_ob;
    int im_ob_p;

    int i_oblique, i_oblique_plus;
    int i_proc;
    double alpha, beta;
	int num_mesh_transfer = NX * NY * NZ_ / xprocs / yprocs;

	//get x slab
#ifdef _MPI
	MPI_Allgather(uu[0], num_mesh_transfer, MPI_DOUBLE, work_v3[0], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
	MPI_Allgather(uu[1], num_mesh_transfer, MPI_DOUBLE, work_v3[1], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
	MPI_Allgather(uu[2], num_mesh_transfer, MPI_DOUBLE, work_v3[2], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
#else
	Copy_v3(work_v3, uu);
#endif

    for (int i = PREV_NPs[REAL][0]; i < NEXT_NPs[REAL][0]; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            orth2obl(j, i, i_oblique, i_oblique_plus, alpha, beta);
            for (int k = 0; k < NPs[REAL][2]; k++) {
                //im = REALMODE_ARRAYINDEX(i, j, k);
				i_proc = i - PREV_NPs[REAL][0];
                im = REALMODE_ARRAYINDEX(i_proc, j, k);
                im_ob   = REALMODE_ARRAYINDEX(i_oblique, j, k);
                im_ob_p = REALMODE_ARRAYINDEX(i_oblique_plus, j, k);

                //orthogonal grid -> oblique grid
                for(int d = 0; d < DIM; d++){
                    uu[d][im] = beta*work_v3[d][im_ob] + alpha*work_v3[d][im_ob_p];
                }

                //orthogonal coordinates -> oblique coordinates
                //warning: mean shear flow is not removed
                uu[0][im] -= (degree_oblique*uu[1][im]);
            }
        }
    }
}

inline void U_oblique2u(double **uu, const bool &add_mean_flow = true) {
    int im;
    int im_ob;
    int im_p;

    int i_plus, i_oblique;
    double alpha, beta;
    int i_ob_slab;
	int num_mesh_transfer = NX * NY * NZ_ / xprocs / yprocs;

#ifdef _MPI
	MPI_Allgather(uu[0], num_mesh_transfer, MPI_DOUBLE, work_v3[0], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
	MPI_Allgather(uu[1], num_mesh_transfer, MPI_DOUBLE, work_v3[1], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
	MPI_Allgather(uu[2], num_mesh_transfer, MPI_DOUBLE, work_v3[2], num_mesh_transfer, MPI_DOUBLE, OWN_X_COMM);
#else
	Copy_v3(work_v3, uu);
#endif

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            obl2orth(j, i, i_plus, i_oblique, alpha, beta);
			if (i_oblique < PREV_NPs[REAL][0] || i_oblique >= NEXT_NPs[REAL][0] ) continue;
            for (int k = 0; k < NPs[REAL][2]; k++) {
                im = REALMODE_ARRAYINDEX(i, j, k);
                //im_ob = REALMODE_ARRAYINDEX(i_oblique, j, k);
                i_ob_slab = i_oblique - PREV_NPs[REAL][0];
                im_ob = REALMODE_ARRAYINDEX(i_ob_slab, j, k);
                im_p    = REALMODE_ARRAYINDEX(i_plus, j, k);

                //oblique grid -> orthogonal grid
                for(int d = 0; d < DIM; d++){
                    //uu[d][im_ob] = beta*work_v3[d][im] + alpha*work_v3[d][im_p];
                    uu[d][im_ob] = beta*work_v3[d][im] + alpha*work_v3[d][im_p];
                }

                uu[0][im_ob] += (degree_oblique*uu[1][im_ob]);
                if(add_mean_flow) uu[0][im_ob] += Shear_rate_eff*((j + PREV_NPs[REAL][1]) - NY/2);
            }
        }
    }
}

//contravariant stress tensor from oblique to orthogonal
inline void Stress_oblique2Stress(double **EE, const bool &add_mean_flow=true){
    int im; 
    int im_ob;
    int im_p;

    int i_plus, i_oblique;
    double alpha, beta;
    int i_ob_slab;

    //double *work_v5[QDIM] = {work_v3[0], work_v3[1], work_v3[2], work_v2[0], work_v2[1]};
    //Copy_v5(work_v5, EE);

#ifdef _MPI
    MPI_Allgather(EE[0], mesh_size, MPI_DOUBLE, work_v3[0], mesh_size, MPI_DOUBLE, OWN_X_COMM);
    MPI_Allgather(EE[1], mesh_size, MPI_DOUBLE, work_v3[1], mesh_size, MPI_DOUBLE, OWN_X_COMM);
    MPI_Allgather(EE[2], mesh_size, MPI_DOUBLE, work_v3[2], mesh_size, MPI_DOUBLE, OWN_X_COMM);
    MPI_Allgather(EE[3], mesh_size, MPI_DOUBLE, work_v4[0], mesh_size, MPI_DOUBLE, OWN_X_COMM);
    MPI_Allgather(EE[4], mesh_size, MPI_DOUBLE, work_v4[1], mesh_size, MPI_DOUBLE, OWN_X_COMM);
#else
	double *work_v5[QDIM] = {work_v3[0], work_v3[1], work_v3[2], work_v2[0], work_v2[1]};
	Copy_v5(work_v5, EE);
#endif

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            obl2orth(j, i, i_plus, i_oblique, alpha, beta);
			if (i_oblique < PREV_NPs[REAL][0] || i_oblique >= NEXT_NPs[REAL][0] ) continue;
            for (int k = 0; k < NPs[REAL][2]; k++) {
                im = REALMODE_ARRAYINDEX(i, j, k);
                i_ob_slab = i_oblique - PREV_NPs[REAL][0];
                im_ob = REALMODE_ARRAYINDEX(i_ob_slab, j, k);
                im_p    = REALMODE_ARRAYINDEX(i_plus, j, k);
            //oblique grid -> orthogonal grid

                EE[0][im_ob] = beta*work_v3[0][im] + alpha*work_v3[0][im_p];
                EE[1][im_ob] = beta*work_v3[1][im] + alpha*work_v3[1][im_p];
                EE[2][im_ob] = beta*work_v3[2][im] + alpha*work_v3[2][im_p];
                EE[3][im_ob] = beta*work_v4[0][im] + alpha*work_v4[0][im_p];
                EE[4][im_ob] = beta*work_v4[1][im] + alpha*work_v4[1][im_p];


            //oblique coordinates -> orthogonal coordinates
            //warning: mean shear flow is added by default!

                //xx
                EE[0][im_ob] += (2.0*degree_oblique*EE[1][im_ob] + SQ(degree_oblique)*EE[3][im_ob]);
                //xy
                EE[1][im_ob] += (degree_oblique*EE[3][im_ob]);
                //xz
                EE[2][im_ob] += (degree_oblique*EE[4][im_ob]);

                if(add_mean_flow) EE[1][im_ob] += (ETA*Shear_rate_eff);
            }//k
        }//j
    }//i
}

inline void Transform_obl_u(double **uu, const OBL_TRANSFORM &flag){

    if(SW_OBL_INT == linear_int){
        if(flag == oblique2cartesian){
            U_oblique2u(uu);
        }else if(flag == cartesian2oblique){
            U2u_oblique(uu);
        }else{
            exit_job(EXIT_FAILURE);
        }
    }else if(SW_OBL_INT == spline_int){
        Spline_u_oblique_transform(uu, flag);
    }else{
        exit_job(EXIT_FAILURE);
    }
}

inline void contra2co(double **contra) {
    int im;

    Copy_v3_k(work_v3, contra);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, contra, work_v3, degree_oblique) private(im)
#endif
    for (int i = 0; i < NPs[SPECTRUM][0]; i++) {
        for (int j = 0; j < NPs[SPECTRUM][1]; j++) {
            for (int k = 0; k < NPs[SPECTRUM][2]; k++) {
                im = SPECTRUMMODE_ARRAYINDEX(i, j, k);
                contra[0][im] = work_v3[0][im] + degree_oblique*work_v3[1][im];
                contra[1][im] = degree_oblique*work_v3[0][im] + (1. + degree_oblique*degree_oblique)*work_v3[1][im];
                contra[2][im] = work_v3[2][im];
            }
        }
    }
}

inline void co2contra(double **contra) {
    int im;

    Copy_v3_k(work_v3, contra);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, contra, work_v3, degree_oblique) private(im)
#endif
    for (int i = 0; i < NPs[SPECTRUM][0]; i++) {
        for (int j = 0; j < NPs[SPECTRUM][1]; j++) {
            for (int k = 0; k < NPs[SPECTRUM][2]; k++) {
                im = SPECTRUMMODE_ARRAYINDEX(i, j, k);
                contra[0][im] = (1. + degree_oblique*degree_oblique)*work_v3[0][im] - degree_oblique*work_v3[1][im];
                contra[1][im] = -degree_oblique*work_v3[0][im] + work_v3[1][im];
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

    contra[0] = dmy[0] + degree_oblique*dmy[1];
    contra[1] = degree_oblique*dmy[0] + (1. + degree_oblique*degree_oblique)*dmy[1];
    contra[2] = dmy[2];
}

inline void co2contra_single(double co[]) {
    double dmy[DIM];

    for (int d = 0; d < DIM; d++) {
        dmy[d] = co[d];
    }

    co[0] = (1. + degree_oblique*degree_oblique)*dmy[0] - degree_oblique*dmy[1];
    co[1] = -degree_oblique*dmy[0] + dmy[1];
    co[2] = dmy[2];
}

/*!
  \brief Compute Fourier transform of scalar field (in place)
  \details \f[A(\vec{r}) \longrightarrow \ft{A}(\vec{k})\f]
  \param[in,out] a scalar field A (input), Fourier transform of A (ouput)
 */

inline void A2a_k(double *a){
    A2a_k_1D (a);  
}

/*!
  \brief Compute inverse Fourier transform of scalar field (in place)
  \details \f[\ft{A}(\vec{k}) \longrightarrow A(\vec{r})\f]
  \param[in,out] a Fourier transform of scalar field A (input), A (ouput)
 */

inline void A_k2a(double *a){
    A_k2a_1D (a);
}

inline void U2u_k (double **u, int dim = DIM) {
    A2a_k_nD (u, dim);
}

inline void U_k2u (double **u, int dim = DIM) {
    A_k2a_nD (u, dim);
}

/*!
  \brief Compute inverse Fourier transform of scalar field
  \details \f[\ft{A}(\vec{k}) \longrightarrow A(\vec{r})\f]
  \param[in] a_k Fourier transform of scalar field A
  \param[out] a_x A in real space
 */

inline void A_k2a_out(double *a_k, double *a_x){ 
    //Copy_v1_k(a_x, a_k);
    int im;
#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, a_x, a_k) private(im)
#endif
    for (int i = 0; i < NPs[SPECTRUM][0]; i++) {
        for (int j = 0; j < NPs[SPECTRUM][1]; j++) {
            for (int k = 0; k < NPs[SPECTRUM][2]; k++) {
                im = SPECTRUMMODE_ARRAYINDEX(i, j, k);
                a_x[im] = a_k[im];
            }
        }
    }
    A_k2a(a_x);
}

/*!
  \brief Compute Fourier transform of scalar field
  \details \f[A(\vec{r}) \longrightarrow \ft{A}(\vec{k})\f]
  \param[in] a_x Real-space scalar field A
  \param[out] a_k Fourier transform of A
 */

inline void A2a_k_out(double *a_x, double *a_k){
    //Copy_v1(a_k, a_x);
    int im;
#ifdef _OPENMP
#pragma omp parallel for default(none) shared (NPs, NZ_, a_x, a_k) private(im)
#endif
    for (int i = 0; i < NPs[REAL][0]; i++) {
        for (int j = 0; j < NPs[REAL][1]; j++) {
            for (int k = 0; k < NPs[REAL][2]; k++) {
                im = REALMODE_ARRAYINDEX(i, j, k);
                a_k[im] = a_x[im];
            }
        }
    }
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

inline void A_k2da_k(double *a, double **da){
    A_k2dxa_k(a,da[0]);
    A_k2dya_k(a,da[1]);
    A_k2dza_k(a,da[2]);
}

inline int Calc_KX_Ooura(const int &i, const int &j, const int &k){
    return (i>HNX) ? i-NX:i;
}

inline int Calc_KY_Ooura(const int &i, const int &j, const int &k){
    assert ( (i < NX) && (j < NY) );
    return (j>HNY) ? j-NY:j;
}

inline int Calc_KZ_Ooura(const int &i, const int &j, const int &k){
    assert ( (i < NX) && (j < NY) );
    return k/2;
}

inline void Truncate_general(double *a, const Index_range &ijk_range){
    Index_range renge;
    if (Range_check (&ijk_range, &renge) ) {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(renge, TRN_Y, TRN_X, TRN_Z, NPs, PREV_NPs, a)
#endif
        for (int i = renge.istart; i <= renge.iend; i++) {
            for (int j = renge.jstart; j <= renge.jend; j++) {
                for (int k = renge.kstart; k <= renge.kend; k++) {
                    assert ( (abs (Calc_KY_Ooura ((i + PREV_NPs[SPECTRUM][0]), (j + PREV_NPs[SPECTRUM][1]), (k + PREV_NPs[SPECTRUM][2]))) >= TRN_Y
                           || abs (Calc_KX_Ooura ((i + PREV_NPs[SPECTRUM][0]), (j + PREV_NPs[SPECTRUM][1]), (k + PREV_NPs[SPECTRUM][2]))) >= TRN_X
                           ||      Calc_KZ_Ooura ((i + PREV_NPs[SPECTRUM][0]), (j + PREV_NPs[SPECTRUM][1]), (k + PREV_NPs[SPECTRUM][2])) >= TRN_Z) );
                    a[SPECTRUMMODE_ARRAYINDEX(i, j, k)] = 0.0;
                }
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
inline void Truncate_two_third_rule_ooura(double *a){
    static Index_range dmy_range;
    const int trn_z2=2*TRN_Z;
    {
        dmy_range.istart=0;
        dmy_range.iend=NX-1;
        dmy_range.jstart=0;
        dmy_range.jend=NY-1;
        dmy_range.kstart=trn_z2;
        dmy_range.kend=NZ_-1;
        Truncate_general(a, dmy_range);
    }
    {
        dmy_range.istart=0;
        dmy_range.iend=NX-1;
        dmy_range.jstart=TRN_Y;
        dmy_range.jend=NY-TRN_Y;
        dmy_range.kstart=0;
        dmy_range.kend=trn_z2-1;
        Truncate_general(a, dmy_range);
    }
    {
        dmy_range.istart=TRN_X;
        dmy_range.iend=NX-TRN_X;
        dmy_range.jstart=0;
        dmy_range.jend=TRN_Y-1;
        dmy_range.kstart=0;
        dmy_range.kend=trn_z2-1;
        Truncate_general(a, dmy_range);
    }
    {
        dmy_range.istart=TRN_X;
        dmy_range.iend=NX-TRN_X;
        dmy_range.jstart=NY-TRN_Y+1;
        dmy_range.jend=NY-1;
        dmy_range.kstart=0;
        dmy_range.kend=trn_z2-1;
        Truncate_general(a, dmy_range);
    }
}
#endif
