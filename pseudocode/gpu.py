

def gpumonty():
	"""
	Pseudocode for the host part of grmonty for the GPU.
	"""

	# initialize random number generator in parallel
	init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */

	# reads data from GRMHD simulation
	init_model(argv);

	# sends simulation data to device
	send_data_to_device(GRMHD)	

	"""
	run kernel that generates (make_super_photon) and solves 
	geodesics (track_super_photon) in parallel
	• dSED corresponds to the SED differential contributed
	   by each unabsorbed photon, returned to the host
	• it should be straightforward to generate several photons
	   in each workitem

	n=1e9 photons would require ~32 GB of host RAM memory,
	therefore this approach would be limited to about a few 1E8 photons.
	It is straightforward to write a loop that repeats the operations
	below, consistent with the amount of RAM available.
	"""
	#kernel make_track_photon(X0[α][n], K0[α][n], dSED[n])
	kernel make_track_photon(dSED[n])

	# assemble SED
	omp_reduce_spect(dSED)
	report_spectrum()







def kernel track_super_photon(X0_α, K0_α, dSED):
	'''
Pseudocode for the kernel that performs geodesic integration of 
one single photon. Includes tracking, absorbing and scattering 
superphotons.
	'''

	# compute covariant metric Gcov at given point in world line
	Gcov=gcov_func(X0) 

	# computes flow properties at X_α—electron density, temperature, B, 
	# velocities from GRMHD data
	Ne, Thetae, B, Ucon, Ucov, Bcon,Bcov = get_fluid_params()

	# computes opacities at given point
	nu = get_fluid_nu(ph->X, ph->K, Ucov);
	alpha_scatti = alpha_inv_scatt(nu, Thetae, Ne);
	alpha_absi = alpha_inv_abs(nu, Thetae, Ne, B, theta);

	# ?????
	dKdlam = init_dKdlam()

	while(integrates geodesic):
		# keeps initial position/wave vector for reference
		Xi = X0
		Ki = K0
		E0 = initial energy

		# integration stepsize along geodesic
		dl=stepsize()

		# advances world line integration one step, computes connection tensor
		X=push_photon()

		# get metric and fluid properties at new position just calculated
		Gcov=gcov_func(X) 
		Ne, Thetae, B, Ucon, Ucov, Bcon,Bcov = get_fluid_params()

		# now computes whether photon is IC-scattered or SS-absorbed 
		if (alpha_absi > 0. or alpha_scatti > 0. or Ne > 0.):
			computes relevant absorption and scattering processes

			# scattering if necessary
			push_photon(Xi, Ki, dKi, dl * frac, &E0, 0);

			# photon ends absorbed

	# records photon on SED
	record_super_photon(ph)


