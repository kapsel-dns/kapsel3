/*!
  \file input.h
  \brief Read udf input file to start simulation (header file)
  \author Y. Nakayama
  \date 2006/11/14
  \version 1.3
 */
#ifndef INPUT_H
#define INPUT_H

#include <assert.h>
#include <stdio.h>
#include <sys/types.h>

#include <cfloat>
#include <cstring>
#include <iostream>
#include <string>

#include "alloc.h"
#include "ewald_wrapper.h"
#include "macro.h"
#include "parameter_define.h"
#include "udfmanager.h"
#include "variable.h"
/////////////////////
/////////////////////
extern int Fixed_particle;
/////////////////////
/////////////////////

enum Particle_IC { None, uniform_random, random_walk, FCC, BCC, user_specify };
enum Particle_IO { random_dir, space_dir, user_dir };
enum SW_time { AUTO, MANUAL };
enum EQ {
    Navier_Stokes,
    Shear_Navier_Stokes,
    Shear_Navier_Stokes_Lees_Edwards,
    Electrolyte,
    Navier_Stokes_FDM,
    Navier_Stokes_Cahn_Hilliard_FDM,
    Shear_Navier_Stokes_Lees_Edwards_FDM,
    Shear_NS_LE_CH_FDM
};
enum ST { explicit_scheme, implicit_scheme };
enum PO { Landau, Flory_Huggins };
enum PT { spherical_particle, chain, rigid };
enum JAX { x_axis, y_axis, z_axis, no_axis };
enum JP { motor, slip, obstacle, no_propulsion };
enum WALL { NO_WALL, FLAT_WALL };
enum QUINCKE { QUINCKE_OFF, QUINCKE_ON };
enum MULTIPOLE { MULTIPOLE_OFF, MULTIPOLE_ON };
enum OBL_INT { linear_int, spline_int };
enum OBL_TRANSFORM { oblique2cartesian, cartesian2oblique };

enum OUTFORMAT { OUT_NONE, OUT_AVS_ASCII, OUT_AVS_BINARY, OUT_EXT };
enum EXTFORMAT { EXT_OUT_HDF5 };

//////
extern SW_time SW_TIME;
//////
extern EQ          SW_EQ;
extern const char *EQ_name[];
//////
extern ST          SW_NSST;
extern ST          SW_CHST;
extern PO          SW_POTENTIAL;
extern const char *NS_SOLVERTYPE_name[];
extern const char *CH_SOLVERTYPE_name[];
extern const char *POTENTIAL_name[];
extern int         PHASE_SEPARATION;
extern int         VISCOSITY_CHANGE;
extern double      eps_ns;
extern int         maxiter_ns;
extern double      eps_ch;
extern int         maxiter_ch;
extern double      XYaspect;
extern double      ETA_A;
extern double      ETA_B;

struct gl_param {
    double a;
    double b;
};
struct fh_param {
    double na;
    double nb;
    double chi;
};
struct ps_param {
    double ratio;
    double init_fluct;
    double d;
    double w;
    double z;
    double alpha;
    double kappa;
    double neutral;
};
extern gl_param gl;
extern fh_param fh;
extern ps_param ps;

//////
extern PT          SW_PT;
extern const char *PT_name[];

//////
extern OBL_INT     SW_OBL_INT;
extern const char *OBL_INT_name[];

//////
extern WALL        SW_WALL;
extern const char *WALL_name[];

//////
extern QUINCKE     SW_QUINCKE;
extern const char *QUINCKE_name[];

//////
extern MULTIPOLE SW_MULTIPOLE;

//////
extern OUTFORMAT    SW_OUTFORMAT;
extern EXTFORMAT    SW_EXTFORMAT;
extern Field_crop   print_field_crop;
extern Field_output print_field;

extern const char *OUTFORMAT_name[];
extern const char *EXTFORMAT_name[];

//////  material parameters
extern double RHO;
extern double IRHO;
extern double ETA;
extern double kBT;
extern double ikBT;
extern double Shear_rate;
extern double Shear_rate_eff;
extern double Shear_strain_realized;
// AC
extern double  Shear_frequency;
extern double  Inertia_stress;
extern double  dev_shear_stress[];
extern double  rigid_dev_shear_stress[];
extern double &dev_shear_stress_lj;
extern double &dev_shear_stress_rot;
extern double &rigid_dev_shear_stress_lj;
extern double &rigid_dev_shear_stress_rot;
extern double  Delta_ETA;
extern double  Nu_ratio;
extern double  NU;
extern double *MASS_RATIOS;
extern double *S_surfaces;  // pretilt scalar order
extern double *W_surfaces;  // spring cst. of anchoring
extern double  IMASS_RATIO;
extern double *IMASS_RATIOS;
extern double *RHO_particle;
extern double *MASS;
extern double *MOI;
extern double *IMASS;
extern double *IMOI;
extern double  EPSILON;
extern double  T_LJ;
extern int     LJ_powers;
extern int     RESUMED;
extern int     last_ts;
extern double  Srate_depend_LJ_cap;
extern double  LJ_dia;

extern int      SW_JANUS;
extern int      SW_JANUS_MOTOR;
extern int      SW_JANUS_SLIP;
extern int      MAX_SLIP_ITER;
extern double   MAX_SLIP_TOL;
extern JAX *    janus_axis;
extern JP *     janus_propulsion;
extern double **janus_force;
extern double **janus_torque;
extern double * janus_slip_vel;
extern double * janus_slip_mode;
extern double * janus_rotlet_C1;
extern double * janus_rotlet_dipole_C2;

//////
extern FlatWall wall;
//////
extern QuinckeEffect quincke;
//////
extern double * multipole_q;   // per species charge
extern double **multipole_mu;  // per species dipole (in body frame)
/*!
    \brief Compute particle dipole in space frame
*/
extern void compute_particle_dipole_standard(double *mu_space, const double *mu_body, quaternion &q);

/*!
    \brief Compute particle dipole in space frame for the special case of a Quincke roller
    \details Assume that the particle dipole is always perpendicular to the plane formed by the anqular velocity vector
   \f$\vec{e}_{\omega}\f$ and the electric field direction \f$\vec{n}\f$

   \f{align*}{
     \vec{p} &=   p_0 \vec{n}\times\vec{e}_{\omega}
   \f}

   with \f$p_0\f$ the magnitude. It is thus parallel to the quincke torque \f$\tau_{\textrm{Quincke}}\f$.

    \warning We are assuming that the magnitude of the dipole is passed in the first component of mu_body
    \param[out] mu_space particle dipole in the lab frame
    \param[in]  mu_body the magnitude of the dipole (p_0, 0, 0)
    \param[in]  q orientation quaternion
*/
extern void compute_particle_dipole_quincke(double *mu_space, const double *mu_body, quaternion &q);
extern void compute_particle_dipole_quincke_image(double *mu_space, const double *mu_body, quaternion &q);

/*!
    \brief Generic funtion pointer used to compute particle dipole
*/
extern void (*compute_particle_dipole)(double *mu_space, const double *mu_body, quaternion &q);
extern void (*compute_particle_dipole_image)(double *mu_space, const double *mu_body, quaternion &q);

//////
extern char Out_dir[];
extern char Out_name[];
//////
extern int SW_UDF;

/////// 計算条件の設定
extern int  Nmax;
extern int  Nmin;
extern int  Ns[];
extern int  Ns_shear[];
extern int  HNs[];
extern int  N2s[];
extern int  HN2s[];
extern int  TRNs[];
extern int  TRNs_QS[];
extern int &NX;
extern int &NY;
extern int &NZ;
extern int  HNZ_;
extern int  NZ_;
extern int  HN2Z_;
extern int  N2Z_;
extern int &HNX;
extern int &HNY;
extern int &HNZ;
extern int &N2X;
extern int &N2Y;
extern int &N2Z;
extern int &HN2X;
extern int &HN2Y;
extern int &HN2Z;
extern int &TRN_X;
extern int &TRN_Y;
extern int &TRN_Z;
extern int &TRN_QS_X;
extern int &TRN_QS_Y;
extern int &TRN_QS_Z;

//////
extern int         ROTATION;
extern int         LJ_truncate;
extern Particle_IC DISTRIBUTION;
extern Particle_IO ORIENTATION;
extern int         N_iteration_init_distribution;
extern int         FIX_CELL;
extern int         FIX_CELLxyz[DIM];
extern int         PINNING;
extern int         N_PIN;
extern int *       Pinning_Numbers;
extern int         N_PIN_ROT;
extern int *       Pinning_ROT_Numbers;
//////
extern double DX;
extern double DX3;
extern double A_XI;
extern double XI;
extern double HXI;
extern double A;
extern double RADIUS;
extern double SIGMA;
extern double A_R_cutoff;
extern double R_cutoff;
//////
extern double G;
extern int    G_direction;
//////
extern int    Component_Number;
extern int    Particle_Number;
extern int *  Particle_Numbers;
extern int *  Beads_Numbers;
extern int *  Chain_Numbers;
extern double VF;
extern double VF_LJ;
extern double Ivolume;
////
extern int       Rigid_Number;
extern int **    Rigid_Motions_vel;    // 0 (fix) or 1 (free)
extern int **    Rigid_Motions_omega;  // 0 (fix) or 1 (free)
extern double ** Rigid_Velocities;
extern double ** Rigid_Omegas;
extern int *     RigidID_Components;
extern int *     Rigid_Particle_Numbers;
extern int *     Rigid_Particle_Cumul;
extern double ** xGs;
extern double ** xGs_previous;
extern double ** xGs_nopbc;
extern double *  Rigid_Masses;
extern double *  Rigid_IMasses;
extern double ***Rigid_Moments;
extern double ***Rigid_IMoments;
extern double ***Rigid_Moments_body;
extern double ** velocityGs;
extern double ** omegaGs;
extern double ** forceGs;   // hydro
extern double ** forceGrs;  // LJ
extern double ** torqueGs;
extern double ** torqueGrs;
extern double ** velocityGs_old;
extern double ** omegaGs_old;
extern double ** forceGs_previous;
extern double ** forceGrs_previous;
extern double ** torqueGs_previous;
extern double ** torqueGrs_previous;
extern int *     Particle_RigidID;
//////
////
extern double **GRvecs;
extern double **GRvecs_body;
//////
extern int GTS;
extern int Num_snap;
extern int MSTEP;
//////
extern double &LX;
extern double &LY;
extern double &LZ;
extern double &HLX;
extern double &HLY;
extern double &HLZ;
extern double  L[];
extern double  HL[];
extern double  L_particle[];
extern double  iL_particle[];
extern double  HL_particle[];
extern double  WAVE_X;
extern double  WAVE_Y;
extern double  WAVE_Z;
extern double  KMAX2;
//////
extern double Axel;
extern double Tdump;
extern double DT_noise;
extern double DT;
/////// Two_fluid
extern double Mean_Bulk_concentration;
extern int    N_spec;
extern double Onsager_solute_coeff;
/////// Electrolyte
extern int     Poisson_Boltzmann;
extern int     External_field;
extern int     AC;
extern int     Shear_AC;
extern double *Surface_charge;
extern double *Surface_charge_e;
extern double  Elementary_charge;
extern double  Valency_counterion;
extern double  Valency_positive_ion;
extern double  Valency_negative_ion;
extern double  Ion_charge;
extern double  Onsager_coeff_counterion;
extern double  Onsager_coeff_positive_ion;
extern double  Onsager_coeff_negative_ion;
extern double  Dielectric_cst;
extern double  Debye_length;
extern double  E_ext[DIM];
extern double  Frequency;
extern double  Angular_Frequency;
////// Temp Monitor

extern double kT_snap_v;  // fix R4A2
extern double kT_snap_o;  // fix R4A2

extern double alpha_v;
extern double alpha_o;

//////shear_degree
extern double degree_oblique;

//////

extern char *      In_udf, *Sum_udf, *Out_udf, *Def_udf, *Ctrl_udf, *Res_udf;
extern UDFManager *ufin, *ufout, *ufres;
// extern UDFManager *ufsum;
/*!
  \brief Read udf files from command line prompt
 */
void file_get(const int argc, char *argv[]);

/*!
  \brief Read udf input file
 */
void Gourmet_file_io(const char *infile,
                     const char *outfile,
                     const char *sumfile,
                     const char *deffile,
                     const char *ctrlfile,
                     const char *resfile);

#endif
