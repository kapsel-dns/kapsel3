#include "operate_surface.h"
void Make_f_slip_particle(double const* const* u,
			  double **up,
			  Particle *p){
  //////////////////////////////////
  const double dx = DX;
  const int np_domain = NP_domain;
  int const* const* sekibun_cell = Sekibun_cell;
  const int * Nlattice = Ns;
  const double radius = RADIUS;
  static const double dmy0 = DX3*RHO;
  //////////////////////////////////

  double xp[DIM], vp[DIM], vp_old[DIM], omega_p[DIM], v_rot[DIM], delta_v[DIM];
  double r[DIM], x[DIM], residue[DIM];
  int x_int[DIM], r_mesh[DIM];
  double dmy_r, dmy_phi;
  int sw_in_cell, pspec;

  double SM[DIM][DIM], Vv[DIM], Sv[DIM];
  double n_r[DIM], n_theta[DIM], n_tau[DIM];
  double polar_axis[DIM], force[DIM], torque[DIM];
  double slip_vel, slip_mode, dmy_xi, dmy_theta, dmy_tau, dmy_slip;
  double dmy_cs, slip_scale, cmass, slip_debug;

#pragma omp parallel for schedule(dynamic, 1)\
  private(xp, vp, omega_p, v_rot, delta_v, r, x, residue, x_int, r_mesh, dmy_r, dmy_phi, \
	  sw_in_cell, pspec, SM, Vv, Sv, n_r, n_theta, n_tau, polar_axis, force, torque, \
	  slip_vel, slip_mode, dmy_xi, dmy_theta, dmy_tau, dmy_slip, dmy_cs, slip_scale, cmass)
  for(int n = 0; n < Particle_Number; n++){

    pspec = p[n].spec;
    if(janus_propulsion[pspec] == slip){

      slip_vel = janus_slip_vel[pspec];
      slip_mode = janus_slip_mode[pspec];
      Janus_direction(polar_axis, p[n]);
      for(int d = 0; d < DIM; d++){
	xp[d] = p[n].x[d];
	vp[d] = p[n].v[d];
	omega_p[d] = p[n].omega[d];
      }
      sw_in_cell = Particle_cell(xp, dx, x_int, residue);
      sw_in_cell = 1;
      fprintf(stderr, "%10.8g %10.8g %10.8g %4d %4d %4d %10.8g %10.8g %10.8g\n",
	      xp[0], xp[1], xp[2],
	      x_int[0], x_int[1], x_int[2],
	      residue[0], residue[1], residue[2]);
      double dmy_fancy[DIM], dmy_brute[DIM];
      {// Compute particle slip velocity correction
	dmy_cs = 0.0;
	for(int d = 0; d < DIM; d++){
	  cmass = 0.0;
	  Vv[d] = Sv[d] = 0.0;
	  SM[d][0] = SM[d][1] = SM[d][2] = 0.0;
	  dmy_fancy[d] = dmy_brute[d] = 0.0;
	}

	for(int mesh = 0; mesh < np_domain; mesh++){
	  Relative_coord(sekibun_cell[mesh], x_int, residue, sw_in_cell, Nlattice, dx, r_mesh, r);
	  int im = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2];
	  
	  for(int d = 0; d < DIM; d++){
	    x[d] = r_mesh[d] * dx;
	  }
	  dmy_r = Distance(x, xp);
	  dmy_phi = 1.0 - Phi(dmy_r, radius);
	  cmass += (1.0 - dmy_phi);
	  dmy_xi = ABS(dmy_r - radius);
	  if(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] > 0){
	    Spherical_coord(r, n_r, n_theta, n_tau, dmy_r, dmy_theta, dmy_tau, p[n]);	  
	    for(int d = 0; d < DIM; d++){
	      dmy_brute[d] += n_r[d];
	      dmy_fancy[d] += r[d];
	    }
	  }
	  if(dmy_xi <= HXI && dmy_phi > 0.0){ // interface domain only
	    Angular2v(omega_p, r, v_rot);

	    
	    dmy_slip = slip_vel * sin(dmy_theta);
	    dmy_cs -= dmy_phi * dmy_slip * sin(dmy_theta);
	    dmy_slip += slip_vel * slip_mode * sin(2.0 * dmy_theta);

	    for(int d = 0; d < DIM; d++){
	      delta_v[d] = (vp[d] + v_rot[d]) - u[d][im];
	    }
	    
	    for(int i = 0; i < DIM; i++){
	      Vv[i] += dmy_phi * n_theta[i] * 
		(delta_v[0] * n_theta[0] + delta_v[1] * n_theta[1] + delta_v[2] * n_theta[2]);
	      Sv [i] += dmy_phi * n_theta[i] * slip_vel;
	      for(int j = 0; j < DIM; j++){
		SM[i][j] += dmy_phi * n_theta[i] * n_theta[j];  
	      }
	    }
	  } // interface domain
	}// mesh
	fprintf(stderr, "%10.8g %10.8g %10.8g / %10.8g / %10.8g / %10.8g  / %10.8g / %10.8g / %10.8g\n",
		xp[0], xp[1], xp[2],
		dmy_brute[0], dmy_fancy[0],
		dmy_brute[1], dmy_fancy[1],
		dmy_brute[2], dmy_fancy[2]);
	M_v_prod(dmy_fancy, SM, vp);

	if(!Fixed_particle){
	  cmass *= dmy0;
	  slip_scale = (-8.0*M_PI/3.0*slip_vel*radius*radius) / (dx*dx*dmy_cs);
	  //	  slip_debug = slip_scale;
	  //	  slip_scale = 1.0;
	  M_scale(SM, dmy0/cmass);
	  for(int d = 0; d < DIM; d++){
	    Vv[d] += slip_scale * Sv[d];
	    SM[d][d] += 1.0;
	  }
	  M_inv(SM);
	  M_v_prod(force, SM, Vv);
	  for(int d = 0; d < DIM; d++){
	    // eff_mass_ratio ?
	    force[d] *= (dmy0/cmass);
	    torque[d] = 0.0;
	    vp_old[d] = vp[d];

	    p[n].f_slip_previous[d] = force[d];
	    p[n].torque_slip_previous[d] = 0.0;
	    p[n].v[d] -= IMASS[pspec] * force[d];
	  }
	}

      }// slip velocity correction

      {// Compute fluid slip velocity 
	for(int d = 0; d < DIM; d++){ // read updated (slip) particle velocities
	  vp[d] = p[n].v[d];
	  omega_p[d] = p[n].omega[d];
	  Vv[d] = Sv[d] = 0.0;
	}
	for(int mesh = 0; mesh < np_domain; mesh++){
	  Relative_coord(sekibun_cell[mesh], x_int, residue, sw_in_cell, Nlattice, dx, r_mesh, r);
	  int im = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2];
	  for(int d = 0; d < DIM; d++){
	    x[d] = r_mesh[d] * dx;
	  }
	  dmy_r = Distance(x, xp);
	  dmy_phi = 1.0 - Phi(dmy_r, radius);
	  dmy_xi = ABS(dmy_r - radius);

	  if(dmy_xi <= HXI && dmy_phi > 0.0){ //interface domain
	    Angular2v(omega_p, r, v_rot);
	    Spherical_coord(r, n_r, n_theta, n_tau, dmy_r, dmy_theta, dmy_tau, p[n]);
	    for(int d = 0; d < DIM; d++){
	      delta_v[d] = (vp[d] + v_rot[d]) - u[d][im];
	    }
	    dmy_slip = slip_scale * slip_vel * (sin(dmy_theta) + slip_mode * sin(2.0 * dmy_theta))
	      + (delta_v[0] * n_theta[0] + delta_v[1] * n_theta[1] + delta_v[2] * n_theta[2]);
	    for(int d = 0; d < DIM; d++){ // careful with parallel update
	      up[d][im] += dmy_slip * dmy_phi * n_theta[d];
	    }
	  }// interface domain

	  // Check momentum conservation over whole fluid domain (eff_mass_ratio?)
	  for(int d = 0; d < DIM; d++){
	    Vv[d] += (1.0 - dmy_phi) * (vp[d] - vp_old[d]);
	    Sv[d] += up[d][im];
	  }
	} // mesh
      }

      for(int d = 0; d < DIM; d++){
	Vv[d] -= Sv[d];
      }
      double dmyv = sqrt(SQ(vp_old[0]) + SQ(vp_old[1]) + SQ(vp_old[2]));
      dmyv = (dmyv > 0 ? 1.0 / dmyv : 1.0);
      // fprintf(stderr, "%10.8g %10.8g %10.8g %10.8g %10.8g %10.8g %10.8g %10.8g\n",
      // 	      slip_debug,
      // 	      ABS(vp[0] - vp_old[0]) * dmyv,
      // 	      ABS(vp[1] - vp_old[1]) * dmyv,
      // 	      ABS(vp[2] - vp_old[2]) * dmyv,
      // 	      sqrt(Vv[0]*Vv[0] + Vv[1]*Vv[1] + Vv[2]*Vv[2]),
      // 	      ABS(Vv[0]), ABS(Vv[1]), ABS(Vv[2])
      // 	      );

    }// slip particle ?
  }// Particle Number
}