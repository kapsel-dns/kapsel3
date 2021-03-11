#ifndef WALL_H
#define WALL_H
#include <assert.h>

#include "input.h"
#include "profile.h"

void Init_Wall(double* phi);
void Init_bottom_Wall(double* phi_wall_prime, double* grad_phi_wall_prime);
void Init_top_Wall(double* phi_wall_double_prime);
void Make_phi_wall_double_prime(double* phi_wall_double_prime);
void Add_f_wall(Particle* p);
#endif
