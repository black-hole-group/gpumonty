

def gpumonty():
	"""
	Pseudocode for the host part of grmonty for the GPU.
	"""

	# initialize random number generator 
	init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */

	# reads data from GRMHD simulation
	init_model(argv);

	# sends fluid simulation data to device
	send_fluid_data_to_device(WHICH VARIABLES???)	

	"""
	Generates photons on the host (make_super_photon) and performs
	radiative transfer (track_super_photon) on the device in parallel.

	n=1e9 photons would require ~32 GB of host RAM memory,
	Therefore the loop below keeps generating photons until the monte carlo
	criteria are satisfied.
	"""
	while (nph_created < nph_desired):

		# generates as many photons as the RAM can hold in the host.
		# Below, SPH is an array of structures with the initial conditions for
		# the photons. 
		# 
		# For example, SPH holds the relevant 4-vectors:
		# • component 1, initial position for the i-th photon SPH[i].X0[1] 
		# • component 3, initial  momentum for the i-th photon SPH[i].K0[3] 
		make_super_photon(SPH)

		# send photon ICs to device
		send_photons_to_device(SPH)

		# run kernel on device to carry out radiative transfer for all photons.
		# Returns histogram of photon energies (SED) `Ener`
		kernel_track_photon(SPH, Ener)

		# clear memory in CPU and GPU
		clear_memory()

		nph_created += nph_gpu

	# assemble SED
	omp_reduce_spect(Ener)
	report_spectrum()







def kernel_track_super_photon(SPH, Ener):
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


