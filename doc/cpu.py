
def grmonty():
	"""
	Pseudocode for grmonty.
	"""

	# initialize random number generator in parallel
	init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */

	# reads data from GRMHD simulation, define global variables
	init_model(argv);

	# initializes number of photons produced, scattered and recorded
	# in the SED 
	N_superph_made = 0;
	N_superph_recorded = 0;
	N_scatt = 0;
	quit_flag = 0;

	# parallel while loop that generates light rays
	while (1):
		if (!quit_flag):
			# generate photon
			make_super_photon(&ph, &quit_flag);
		
		# if critical number of photons were produced, finish
		if (quit_flag):
			break;

		# solve geodesics
		track_super_photon(&ph);

		N_superph_made += 1;

	# Once done with all photons, assemble SED
	omp_reduce_spect();
	report_spectrum((int) N_superph_made);











def track_super_photon(initial photon X0_α and K0_α, generated 
	outside function):
	'''
Pseudocode for the geodesic integration of one single geodesic.

Includes tracking, absorbing and scattering superphotons.
	'''

	# compute covariant metric Gcov at given point in world line
	Gcov=gcov_func(X) 

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
		Xi = initial position
		Ki = initial wave 4-vector
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
















