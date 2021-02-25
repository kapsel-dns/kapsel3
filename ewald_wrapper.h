/*!
  \file ewald_wrapper.h
  \author John J. Molina
  \date 2014/08/18
  \version 1.0
  \brief Wrapper to compute ewald forces, fields, field gradients, etc.
 */

#ifndef EWALD_WRAPPER_H
#define EWALD_WRAPPER_H

#include <stdio.h>
#include <time.h>

#include "ewald.h"

typedef struct EwaldParams {
    // input parameters
    double alpha   = 8.0;
    double delta   = EPSILON_MP;
    double conv    = 0.51;
    double epsilon = -1.0;  // tinfoil if < 0
    double Pz_factor;
    bool   charge  = false;
    bool   dipole  = false;
    bool   enabled = false;
    bool   m_image = false;
    void   init(double _alpha, double _delta, double _conv, double _eps, bool _charge, bool _dipole) {
        alpha   = _alpha;
        delta   = _delta;
        conv    = _conv;
        epsilon = _eps;
        charge  = _charge;
        dipole  = _dipole;
        enabled = _charge || _dipole;
    }
} EwaldParams;
extern EwaldParams ewald_param;

typedef struct EwaldMem {
    int  num      = 0;
    int *group_id = nullptr;

    double **r  = nullptr;
    double * q  = nullptr;
    double **mu = nullptr;

    double ** force;
    double ** torque;
    double ** efield;
    double ***efield_grad;
    double    energy[5];

    /*!
    \brief Initialize auxiliary Ewald arrays
    \warning ewald_param should be initilized before calling this function!!!
    */
    void init(const int &_num) {
        num      = _num;
        group_id = (int *)alloc_1d_int(num);
        if (ewald_param.m_image == true) {
            r        = (double **)alloc_2d_double(num*2, DIM);
            mu       = (ewald_param.dipole ? (double **)alloc_2d_double(num*2, DIM) : nullptr);
        } else {
            r        = (double **)alloc_2d_double(num, DIM);
            mu       = (ewald_param.dipole ? (double **)alloc_2d_double(num, DIM) : nullptr);
        }
        q        = (ewald_param.charge ? (double *)alloc_1d_double(num) : nullptr);

        if (ewald_param.m_image == true) {
            {
                double *rr = r[0];
                for (int i = 0; i < num * 2 * DIM; i++) rr[i] = 0.0;
                if (ewald_param.dipole) {
                    double *pp = mu[0];
                    for (int i = 0; i < num * 2 * DIM; i++) pp[i] = 0.0;
                }
            }
        } else {
            {
                double *rr = r[0];
                for (int i = 0; i < num * DIM; i++) rr[i] = 0.0;
                if (ewald_param.dipole) {
                    double *pp = mu[0];
                    for (int i = 0; i < num * DIM; i++) pp[i] = 0.0;
                }
            }
        }
        for (int i = 0; i < num; i++) group_id[i] = -(i + 1);
                if (ewald_param.charge) {
                    for (int i = 0; i < num; i++) q[i] = 0.0;
                }

        force       = (double **)alloc_2d_double(num, DIM);
        torque      = (double **)alloc_2d_double(num, DIM);
        efield      = (double **)alloc_2d_double(num, DIM);
        efield_grad = (double ***)alloc_3d_double(num, DIM, DIM);
    }
    void free() {
        free_1d_int(group_id);

        if (ewald_param.charge) free_1d_double(q);

        free_2d_double(r);
        if (ewald_param.dipole) free_2d_double(mu);

        free_2d_double(force);
        free_2d_double(torque);
        free_2d_double(efield);
        free_3d_double(efield_grad);
    }
} EwaldMem;
extern EwaldMem ewald_mem;

extern parallelepiped *ewald_cell;
extern ewald *         ewald_sum;

void free_ewald_sum();

void init_ewald_sum(const double &lx, const double &ly, const double &lz, const int &num);
void compute_ewald_sum();
void print_ewald_info(FILE *stream);

#endif
